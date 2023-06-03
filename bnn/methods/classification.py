from typing import Optional, Callable
import pyro.optim

import torch
from ml_collections import ConfigDict
from pyro.infer import SVI, Predictive, Trace_ELBO, JitTrace_ELBO
from pyro.infer.mcmc import NUTS, MCMC
from pyro.nn import PyroModule
from torchmetrics import F1Score, MeanMetric, Accuracy, Precision, Recall
from tqdm import tqdm

from bnn.utils.dataset import get_dataset, create_dataloaders
from bnn.utils.guide_factory import get_guide
from bnn.utils.model_factory import get_classifier_model
from bnn.utils.optim_factory import get_optim
from bnn.utils.torch_utils import get_lr, init_svi_method, save_mcmc_checkpoint, save_svi_checkpoint



def classification_svi_method(config: ConfigDict, log: Callable, checkpoint: Optional[str] = None):
    train_dataset, val_dataset = get_dataset(config)
    model = get_classifier_model(config)
    model = model.to(config.device)
    guide = get_guide(config, model)
    guide = guide.to(config.device)
    optim = get_optim(config)
    loss = JitTrace_ELBO(**config.loss_config) if config.loss_jit else Trace_ELBO(**config.loss_config)
    infer = SVI(model, guide, optim, loss=loss)
    epoch, step = init_svi_method(model, optim, guide, checkpoint)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, config)
    step = epoch * len(train_dataloader)  # TODO fix step checkpointing
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
            log_dict = {'train/loss': loss}
            
            log(log_dict, step)
            if step % config.pbar_step == 0:
                pbar.set_description(f'Training Epoch: {epoch}, loss={loss:.4f}')
            step += 1
        # val step
        guide.requires_grad_(False)
        model.eval()
        guide.eval()
        f1_score = F1Score('multiclass', num_classes=config.num_classes)
        accuracy = Accuracy('multiclass', num_classes=config.num_classes)
        perc = Precision('multiclass', num_classes=config.num_classes)
        recall = Recall('multiclass', num_classes=config.num_classes)
        val_loss = MeanMetric()
        predictive = Predictive(model, guide=guide, return_sites=['obs'], num_samples=config.predict_num_samples).to(config.device)
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Validation Epoch: {epoch}'):
                data, label = batch
                data = data.to(config.device)
                label = label.to(config.device)
                
                if config.model_name == 'bnn':
                    data = torch.flatten(data, 1)
                loss = infer.evaluate_loss(data, label)
                y_pred = torch.round(torch.mean(predictive(data)['obs'].detach().float(), dim=0))
                f1_score.update(y_pred.cpu(), label.cpu())
                accuracy.update(y_pred.cpu(), label.cpu())
                perc.update(y_pred.cpu(), label.cpu())
                recall.update(y_pred.cpu(), label.cpu())
                val_loss.update(loss)

            val_loss_value = val_loss.compute()
            f1_value = f1_score.compute()
            accuracy_value = accuracy.compute()
            perc_val = perc.compute()
            rec_val = recall.compute()
            print(f'Validation Loss: {val_loss_value:.4f}')
            print(f'Validation F1 Score: {f1_value:.4f}')
            print(f'Validation Accuracy: {accuracy_value:.4f}')
            print(f'Validation Precision: {perc_val:.4f}')
            print(f'Validation Recall: {rec_val:.4f}')
            log({'val/loss': val_loss_value,
                 'val/f1': f1_value,
                 'val/precision': perc_val,
                 'val/recall': rec_val,
                 'val/accuracy': accuracy_value},
                step,
                validation=True)
        epoch += 1
        save_svi_checkpoint(config.checkpoint_root, epoch, step, model, optim, guide)




def classification_mcmc_method(config: ConfigDict, log: Callable, checkpoint: Optional[str] = None):
    train_dataset, val_dataset = get_dataset(config)
    model = get_classifier_model(config)
    model = model.to(config.device)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, config)
    kernel = NUTS(model, **config.nuts_config)
    infer = MCMC(kernel, **config.mcmc_config)
    model.train()
    for batch in train_dataloader:
        data, label = batch
        data = data.to(config.device)
        label = label.to(config.device)
        if config.model_name == 'bnn':
            data = torch.flatten(data, 1)
        
        infer.run(data, label)
    
    
    model.eval()
    f1_score = F1Score('multiclass', num_classes=config.num_classes)
    predictive = Predictive(model=model, posterior_samples=infer.get_samples())
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Validation Epoch: {epoch}'):
            data, label = batch
            data = data.to(config.device)
            label = label.to(config.device)
            if config.model_name == 'bnn':
                data = torch.flatten(data, 1)
            y_pred = torch.round(torch.mean(predictive(data)['obs'].detach().float(), dim=0))
            f1_score.update(y_pred.cpu(), label.cpu())
        f1_value = f1_score.compute()
        print(f'Validation F1 Score: {f1_value:.4f}')
        log({'val/f1': f1_score.compute()},
                None,
                validation=True)
        
    
    
    save_mcmc_checkpoint(config.checkpoint_path, model)