from trainers.base_trainer import BaseTrainer
from trainers.occam_trainer import OccamTrainer
import torch
import torch.nn.functional as F




class cSGMCMCTrainer(BaseTrainer):
    """
    Implementation for:
    cSG-MCMC

    """
    def __init__(self. config):
        super().__init__()

        self.moments = 0

        # turn off optimizer
        if self.trainer_cfg.is_hmc:
            self.automatic_optimization = False


        self.num_batches = self.dataset_cfg.dataset_size / self.dataset_cfg.batch_size + 1
        self.iterations = self.num_batches * self.dataset_cfg.epochs * self.trainer_cfg.cycles
        self.temperature = 1. / self.dataset_cfg.dataset_size
        self.cycle_length = int(self.dataset_cfg.epochs / self.trainer_cfg.models_per_cycle)

    def noise_loss(self, lr):
        noise_loss = 0.0
        noise_std = (2 / lr * self.trainer_cfg.alpha)**0.5
        for var in self.model.parameters():
            means = torch.zeros(var.size()).to(self.device)
            noise_loss += torch.sum(var * torch.normal(means, std=noise_std).to(self.device))

        return noise_loss

    def update_params(self, lr):
        for p in self.model.parameters():
            if not hasattr(p,'buf'):
                p.buf = torch.zeros(p.size()).to(self.device)
            d_p = p.grad.data
            d_p.add_(p.data, alpha=self.trainer_cfg.weight_decay)

            buf_new = (1 - self.trainer_cfg.alpha) * p.buf - lr * d_p
            if (self.current_epoch % self.dataset_cfg.epochs) + 1 > (self.dataset_cfg.epochs - self.trainer_cfg.models_per_cycle):
                eps = torch.randn(p.size()).to(self.device)
                buf_new += (2.0 * self.optim_cfg.lr * self.trainer_cfg.alpha * self.temperature / self.dataset_cfg.datasize)**.5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

    def adjust_learning_rate(self, batch_idx):
        rcounter = self.current_epoch * self.dataset_cfg.num_batches + batch_idx

        cos_inner = np.pi * (rcounter % (self.iterations // self.trainer_cfg.cycles))
        cos_inner /= self.iterations // self.trainer_cfg.cycles
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.optim_cfg.args.lr

        if not self.is_hmc:
            # the cSG-HMC code does not change the lr here, bc does custom param update
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = lr

        return lr

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.validation_step(batch, batch_idx, 'val', dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        # TODO first saave model moments



        # return self.shared_validation_step(batch, batch_idx, 'test', dataloader_idx)


    def validation_step(self, batch, batch_idx, split, dataloader_idx=None, model_out=None):

        if model_out is None:
            model_out = self(batch['x'], batch)
        loader_key = self.get_loader_key(split, dataloader_idx)

        if batch_idx == 0:
            accuracy = Accuracy()
            setattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_accuracy', accuracy)
            self.init_calibration_analysis(split, loader_key)

        accuracy = getattr(self, f'{split}_{self.get_loader_key(split, dataloader_idx)}_accuracy')
        self.accuracy_metric_step(batch, batch_idx, model_out, split, dataloader_idx, accuracy)
        self.segmentation_metric_step(batch, batch_idx, model_out, split, dataloader_idx)
        self.calibration_analysis_step(batch, batch_idx, split, dataloader_idx, model_out)

    def validation_epoch_end(self, outputs):

        if (self.current_epoch % self.cycle_length) + 1 > self.cycle_length - self.trainer_cfg.models_per_cycle:

            print(f"SAVING MOMENT {self.moment}, epoch no. {self.current_epoch}")
            # TODO where and how to save
            torch.save(self.model.state_dict(), f"{MOMENTS_DIR}{self.model_desc}/moment_{self.moment}.pt")

            self.moments += 1


        return self.shared_validation_epoch_end(outputs, 'val')

    def training_step(self, batch, batch_idx):
        if self.is_hmc:
            self.model.zero_grad()

        model_out = self(batch['x'], batch)

        lr = self.adjust_learning_rate(batch_idx)

        if not self.is_hmc:
            if (self.current_epoch % self.dataset_cfg.epochs) + 1 > (self.dataset_cfg.epochs - self.trainer_cfg.models_per_cycle):
                loss_noise = self.noise_loss(lr) * (self.temperature / self.dataset_cfg.dataset_size)**.5
                loss = self.compute_main_loss(batch, batch_idx, model_out) + loss_noise
            else:
                loss = self.compute_main_loss(batch, batch_idx, model_out)
        else:
            # cSG-HMC regular loss
            loss = self.compute_main_loss(batch, batch_idx, model_out)
            self.manual_backward(loss)

            # this is the update
            self.update_params(lr)

        self.log('loss', loss, on_epoch=True, batch_size=self.config.dataset.batch_size, py_logging=False)
        return loss
