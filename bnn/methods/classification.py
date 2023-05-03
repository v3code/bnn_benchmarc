from typing import Optional, Callable

import torch
from ml_collections import ConfigDict
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.nn import PyroModule
from torchmetrics import F1Score, MeanMetric
from tqdm import tqdm

from bnn.utils.dataset import get_dataset, create_dataloaders
from bnn.utils.guide_factory import get_guide
from bnn.utils.model_factory import get_classifier_model
from bnn.utils.optim_factory import get_optim
from bnn.utils.torch_utils import init_svi_method, save_svi_checkpoint



def classification_svi_method(config: ConfigDict, log: Callable, checkpoint: Optional[str] = None):
    train_dataset, val_dataset = get_dataset(config)
    model = get_classifier_model(config)
    model = model.to(config.device)
    guide = get_guide(config, model)
    guide = guide.to(config.device)
    optim = get_optim(config)
    infer = SVI(model, guide, optim, loss=Trace_ELBO(**config.loss_config))
    epoch, step = init_svi_method(model, optim, guide, checkpoint)
    step = 0  # TODO fix step checkpointing
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, config)
    while epoch < config.epochs:
        # train step
        guide.requires_grad_(True)
        model.train()
        guide.train()
        pbar = tqdm(train_dataloader, desc=f'Training Epoch: {epoch}, loss=0')
        for batch in pbar:
            data, label = batch
            data = data.to(config.device)
            label = label.to(config.device)
            if config.model_name == 'bnn':
                data = torch.flatten(data, 1)

            loss = infer.step(data, label)

            log({'train/loss': loss}, step)
            if step % config.pbar_step == 0:
                pbar.set_description(f'Training Epoch: {epoch}, loss={loss:.4f}')
            step += 1
        # val step
        guide.requires_grad_(False)
        model.eval()
        guide.eval()
        f1_score = F1Score('multiclass', num_classes=config.model.num_classes).to(config.device)
        val_loss = MeanMetric().to(config.device)
        predictive = Predictive(model, guide=guide, return_sites=['obs'], num_samples=config.predict_num_samples).to(config.device)
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Validation Epoch: {epoch}'):
                data, label = batch
                data = data.to(config.device)
                label = label.to(config.device)
                loss = infer.evaluate_loss(data, label)
                y_pred = torch.round(torch.mean(predictive(data)['obs'].float(), dim=0)).to(torch.int8)
                f1_score.update(y_pred, label)
                val_loss.update(loss)

            val_loss_value = val_loss.compute()
            f1_value = f1_score.compute()
            print(f'Validation Loss: {val_loss_value:.4f}')
            print(f'Validation F1 Score: {f1_value:.4f}')
            log({'val/loss': val_loss.compute(),
                 'val/f1': f1_score.compute()},
                step,
                validation=True)
        epoch += 1
        save_svi_checkpoint(config.checkpoint_path, epoch, step, model, optim, guide)

def classification_mcmc_method(config: ConfigDict, log: Callable, checkpoint: Optional[str] = None):
    train_dataset, val_dataset = get_dataset(config)
    model = get_classifier_model(config)
    model = model.to(config.device)
    guide = get_guide(config, model)
    guide = guide.to(config.device)
    optim = get_optim(config)
    infer = SVI(model, guide, optim, loss=Trace_ELBO(**config.loss_config))
    epoch, step = init_svi_method(model, optim, guide, checkpoint)
    step = 0  # TODO fix step checkpointing
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, config)
    while epoch < config.epochs:
        # train step
        guide.requires_grad_(True)
        model.train()
        guide.train()
        pbar = tqdm(train_dataloader, desc=f'Training Epoch: {epoch}, loss=0')
        for batch in pbar:
            data, label = batch
            data = data.to(config.device)
            label = label.to(config.device)
            if config.model_name == 'bnn':
                data = torch.flatten(data, 1)

            loss = infer.step(data, label)

            log({'train/loss': loss}, step)
            if step % config.pbar_step == 0:
                pbar.set_description(f'Training Epoch: {epoch}, loss={loss:.4f}')
            step += 1
        # val step
        guide.requires_grad_(False)
        model.eval()
        guide.eval()
        f1_score = F1Score('multiclass', num_classes=config.model.num_classes)
        val_loss = MeanMetric()
        predictive = Predictive(model, guide=guide, return_sites=['obs'], num_samples=config.predict_num_samples)
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Validation Epoch: {epoch}'):
                data, label = batch
                data = data.to(config.device)
                label = label.to(config.device)
                loss = infer.evaluate_loss(data, label)
                y_pred = torch.round(torch.mean(predictive(data)['obs'].float(), dim=0)).to(torch.int8)
                f1_score.update(y_pred, label)
                val_loss.update(loss)

            val_loss_value = val_loss.compute()
            f1_value = f1_score.compute()
            print(f'Validation Loss: {val_loss_value:.4f}')
            print(f'Validation F1 Score: {f1_value:.4f}')
            log({'val/loss': val_loss.compute(),
                 'val/f1': f1_score.compute()},
                step,
                validation=True)
        epoch += 1
        save_svi_checkpoint(config.checkpoint_path, epoch, step, model, optim, guide)