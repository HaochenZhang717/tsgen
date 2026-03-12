import os
import torch
from tqdm.auto import tqdm
import copy

class Trainer:
    def __init__(self, config, args, model, dataloader):
        self.config = config
        self.args = args
        self.model = model

        self.train_dataloader = dataloader["train_dataloader"]
        self.valid_dataloader = dataloader["valid_dataloader"]

        self.device = next(self.model.parameters()).device

        self.base_lr = config["solver"]["base_lr"]
        self.max_epochs = config["solver"]["max_epochs"]
        self.save_cycle = config["solver"]["save_cycle"]

        self.results_folder = os.environ.get(
            "results_folder",
            config["solver"].get("results_folder", "./Checkpoints_default")
        )
        os.makedirs(self.results_folder, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        self.ema_model = copy.deepcopy(model)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.ema_decay = 0.999


    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)


    def save(self, milestone):
        data = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "milestone": milestone,
        }
        torch.save(data, os.path.join(self.results_folder, f"model-{milestone}.pt"))

    def load(self, milestone):
        path = os.path.join(self.results_folder, f"model-{milestone}.pt")
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema_model"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        print(f"Loaded checkpoint from {path}")

    def train(self):
        self.model.train()
        best_val_loss = float("inf")
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            total_num = 0

            self.model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.max_epochs}", leave=False)
            for batch in pbar:
                x = batch.to(self.device)  # train dataloader only returns x
                loss = self.model(x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_ema()

                bs = x.shape[0]
                total_loss += loss.item() * bs
                total_num += bs
                pbar.set_postfix(loss=f"{loss.item():.6f}")


            if epoch % 10 == 0:
                total_loss_val = 0.0
                total_num_val = 0
                pbar = tqdm(self.valid_dataloader, desc=f"Evaluate Epoch {epoch}/{self.max_epochs}", leave=False)
                for batch in pbar:
                    x = batch.to(self.device)  # train dataloader only returns x
                    with torch.no_grad():
                        loss = self.ema_model(x)
                    bs = x.shape[0]
                    total_loss_val += loss.item() * bs
                    total_num_val += bs
                    pbar.set_postfix(loss=f"{loss.item():.6f}")
                avg_loss_val = total_loss_val / max(total_num_val, 1)
                print(f"[Epoch {epoch}] valid loss = {avg_loss_val:.6f}")
                if avg_loss_val < best_val_loss:
                    best_val_loss = avg_loss_val
                    self.save("best")

            avg_loss = total_loss / max(total_num, 1)
            print(f"[Epoch {epoch}] train loss = {avg_loss:.6f}")

            if epoch % self.save_cycle == 0 or epoch == self.max_epochs:
                self.save(epoch)



    @torch.no_grad()
    def restore(self, test_dataloader):
        """
        test_dataloader yields (x, mask)
        x: [B, L, D] in [-1,1]
        mask: boolean, True means observed, False means target to predict
        """
        self.model.eval()

        all_samples = []
        all_reals = []
        for batch in tqdm(test_dataloader):
            x = batch
            x = x.to(self.device)
            sample = self.ema_model.generate_mts(batch_size=x.shape[0])
            all_samples.append(sample.detach().cpu())
            all_reals.append(x.detach().cpu())
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_reals = torch.cat(all_reals, dim=0).numpy()

        dict_to_save = {
            'sample': all_samples,
            'orig': all_reals,
        }

        torch.save(dict_to_save, os.path.join(self.results_folder, f"samples.pt"))

        return all_samples, all_reals