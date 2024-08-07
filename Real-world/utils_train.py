import torch
import numpy as np
import scipy.sparse as sp


def compute_cate_loss(x_hat, prob, matrix_F, prob_mask):
    recon_prob = torch.einsum('ix, jx -> ij', x_hat, matrix_F)
    recon_prob = recon_prob.softmax(dim=-1)
    loss = torch.nn.MSELoss(reduction='none')(recon_prob, prob).mean(dim=-1)
    loss = torch.where(prob_mask == 1, 0, loss).mean(dim=-1)
    return loss


def reweight_loss(w_min, w_max, prob, matrix_F):
    reweight = torch.einsum('ix, xj -> ij', prob, matrix_F)
    min_reweight = reweight.min(dim=-1)[0].unsqueeze(-1)
    max_reweight = reweight.max(dim=-1)[0].unsqueeze(-1)

    return w_min + (w_max - w_min) * (reweight - min_reweight) / (max_reweight - min_reweight)


def compute_recon_loss(args, diffusion, timesteps, pt, x_0_hat, x_0, prob, matrix_F):
    loss_batch_item = ((x_0_hat - x_0) ** 2)

    positive_balance = reweight_loss(args.w_min, args.w_max, 1 - prob, matrix_F)
    negative_balance = reweight_loss(args.w_min, args.w_max, prob, matrix_F)

    balance_weight = torch.where(x_0 == 1, positive_balance, negative_balance)
    loss_batch = (loss_batch_item * balance_weight).mean(dim=-1)

    if args.snr is True:
        weight = 0.5 * (diffusion.SNR(timesteps - 1) - diffusion.SNR(timesteps))
        weight = torch.where((timesteps == 0), 1.0, weight)
    else:
        weight = torch.tensor([1.0] * x_0.shape[0]).to(args.device)

    weighted_loss_batch = weight * loss_batch

    # Update Lt_history & Lt_count for importance sampling
    for timestep, loss in zip(timesteps, weighted_loss_batch):
        if diffusion.Lt_count[timestep] == diffusion.num_for_expectation:
            Lt_history_old = diffusion.Lt_history.clone()
            diffusion.Lt_history[timestep, :-1] = Lt_history_old[timestep, 1:]
            diffusion.Lt_history[timestep, -1] = loss.detach()
        else:
            try:
                diffusion.Lt_history[timestep, diffusion.Lt_count[timestep]] = loss.detach()
                diffusion.Lt_count[timestep] += 1
            except:
                print(timestep)
                print(diffusion.Lt_count[timestep])
                print(loss)
                raise ValueError

    weighted_loss_batch /= pt
    weighted_loss = weighted_loss_batch.mean()

    return weighted_loss


def calculate_loss(args, model, diffusion, x_0, prob, matrix_F):
    timesteps, pt = diffusion.sample_steps(x_0.shape[0], 'importance')
    if args.noise_scale != 0:
        x_t, noise = diffusion.get_noised_interaction(x_0, timesteps)
    else:
        x_t = x_0

    # Drop diversity class with some probability
    prob_mask = torch.bernoulli(torch.zeros(prob.shape[0]) + args.drop_div).to(args.device)
    x_hat_recon, x_hat_prob, x_hat_cate, loss_ortho = model(x_t, timesteps, prob, prob_mask)
    loss_recon = compute_recon_loss(args, diffusion, timesteps, pt, x_hat_recon, x_0, prob, matrix_F)
    loss_cate = compute_cate_loss(x_hat_prob, prob, matrix_F, prob_mask)
    loss_emb = torch.nn.MSELoss()(input=x_hat_cate, target=matrix_F)
    return loss_recon + loss_cate + args.lamda * (loss_ortho + loss_emb)


def train_one_epoch(args, model, diffusion, optimizer, loader, matrix_F):
    model.train()
    total_loss = 0.0

    for x_0, prob, prob_pred in loader:
        x_0 = x_0.to(args.device)
        prob = prob.to(args.device)

        loss = calculate_loss(args, model, diffusion, x_0, prob, matrix_F)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)