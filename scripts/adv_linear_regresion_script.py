# %%
%load_ext autoreload
%autoreload 2
# %%
import os
from os.path import join
import numpy as np
import torch
from tqdm import tqdm, trange
from PIL import Image
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle as pkl
from circuit_toolkit.plot_utils import saveallforms
from time import time
ffhq256 = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256/ffhq256"

def create_circle_mask(img_size, radius=0.3):
    """Create a circular mask in the center of the image."""
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, y)
    circle_mask = (X**2 + Y**2) < radius**2
    return circle_mask.astype(float)

def make_linear_regression_experiment_Xy(n_samples, img_size, noise_level, radius=0.3):
    """Make linear regression experiment data."""
    img_mat_all = torch.from_numpy(np.array(img_collection)).float() / 255.0
    images_mat = img_mat_all[:n_samples]
    ground_truth_mask = create_circle_mask(img_size, radius=radius)
    ground_truth_coef = torch.from_numpy(ground_truth_mask.flatten()).float()
    y_ground_truth = (images_mat.cuda() @ ground_truth_coef.cuda()).cpu()
    return images_mat, y_ground_truth


def run_linear_regression_experiment_Xy(images_mat, y_ground_truth, noise_level=5, run_lasso=False):
    """Run linear regression experiment with different models."""
    n_samples = images_mat.shape[0]
    images_mat = images_mat.reshape(n_samples, -1)
    
    # Add noise to create observed y
    y_observed = y_ground_truth + torch.randn(n_samples) * noise_level
    print("y_ground_truth.std()", y_ground_truth.std())
    print("y_observed.std()", y_observed.std())
    print("noise_level", noise_level)
    # Fit models
    coef_est_OLS = LinearRegression().fit(images_mat, y_observed).coef_
    coef_est_Ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(images_mat, y_observed).coef_
    if run_lasso:
        coef_est_Lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]).fit(images_mat, y_observed).coef_
    
    return {
        'coef_OLS': coef_est_OLS.reshape(img_size, img_size),
        'coef_Ridge': coef_est_Ridge.reshape(img_size, img_size),
        'coef_Lasso': coef_est_Lasso.reshape(img_size, img_size) if run_lasso else None
    }


def run_linear_regression_experiment(images_mat, ground_truth_weight_mask, noise_level=5, run_lasso=False):
    """Run linear regression experiment with different models."""
    n_samples = images_mat.shape[0]
    img_size = images_mat.shape[1]
    images_mat = images_mat.reshape(n_samples, -1)
    # Generate ground truth y using the mask
    ground_truth_coef = torch.from_numpy(ground_truth_weight_mask.flatten()).float()
    y_ground_truth = (images_mat.cuda() @ ground_truth_coef.cuda()).cpu()
    # Add noise to create observed y
    y_observed = y_ground_truth + torch.randn(n_samples) * noise_level
    print("y_ground_truth.std()", y_ground_truth.std())
    print("y_observed.std()", y_observed.std())
    print("noise_level", noise_level)
    images_mat_np = images_mat.cpu().numpy()
    y_observed_np = y_observed.cpu().numpy()
    # Fit models
    # Fit OLS and get R2
    ols_model = LinearRegression().fit(images_mat_np, y_observed_np)
    coef_est_OLS = ols_model.coef_
    r2_ols = ols_model.score(images_mat_np, y_observed_np)
    
    # Fit Ridge and get R2
    ridge_model = RidgeCV(alphas=list(np.logspace(-4, 5, 10))).fit(images_mat_np, y_observed_np)
    coef_est_Ridge = ridge_model.coef_
    r2_ridge = ridge_model.score(images_mat_np, y_observed_np)
    
    # Fit Lasso and get R2 if requested
    if run_lasso:
        lasso_model = LassoCV(alphas=list(np.logspace(-4, 5, 10))).fit(images_mat_np, y_observed_np)
        coef_est_Lasso = lasso_model.coef_
        r2_lasso = lasso_model.score(images_mat_np, y_observed_np)
    
    print(f"R2 scores - OLS: {r2_ols:.3f}, Ridge: {r2_ridge:.3f}" + (f", Lasso: {r2_lasso:.3f}" if run_lasso else ""))
    
    return {
        'ground_truth': ground_truth_weight_mask,
        'coef_OLS': coef_est_OLS.reshape(img_size, img_size),
        'coef_Ridge': coef_est_Ridge.reshape(img_size, img_size),
        'coef_Lasso': coef_est_Lasso.reshape(img_size, img_size) if run_lasso else None,
        'r2_OLS': r2_ols,
        'r2_Ridge': r2_ridge,
        'r2_Lasso': r2_lasso if run_lasso else None
    }

def visualize_results(results, title=""):
    """Visualize the results of the linear regression experiment."""
    figh = plt.figure(figsize=(16, 4.25))
    
    plt.subplot(1, 4, 1)
    plt.imshow(results['ground_truth'], cmap='coolwarm')
    plt.colorbar()
    plt.title('Ground Truth')
    
    plt.subplot(1, 4, 2)
    plt.imshow(results['coef_OLS'], cmap='coolwarm')
    plt.colorbar()
    if 'r2_test_OLS' in results:
        plt.title('OLS\n 1-R2 train={:.2e}\ntest={:.2e}'.format(1-results['r2_train_OLS'], 1-results['r2_test_OLS']))
    else:
        plt.title('OLS\n 1-R2 train={:.2e}'.format(1-results['r2_train_OLS']))
    
    plt.subplot(1, 4, 3)
    plt.imshow(results['coef_Ridge'], cmap='coolwarm')
    plt.colorbar()
    if 'r2_test_Ridge' in results:
        plt.title('Ridge\n 1-R2 train={:.2e}\ntest={:.2e}'.format(1-results['r2_train_Ridge'], 1-results['r2_test_Ridge']))
    else:
        plt.title('Ridge\n 1-R2 train={:.2e}'.format(1-results['r2_train_Ridge']))
    
    plt.subplot(1, 4, 4)
    if results['coef_Lasso'] is not None:
        plt.imshow(results['coef_Lasso'], cmap='coolwarm')
        plt.colorbar()
        if 'r2_test_Lasso' in results:
            plt.title('Lasso\n 1-R2 train={:.2e}\ntest={:.2e}'.format(1-results['r2_train_Lasso'], 1-results['r2_test_Lasso']))
        else:
            plt.title('Lasso\n 1-R2 train={:.2e}'.format(1-results['r2_train_Lasso']))
    else:
        plt.text(0.5, 0.5, 'Lasso not run', ha='center', va='center', fontsize=12)
    plt.suptitle(title)
    plt.show()
    return figh

# Example usage:
# img_size = 100  # or whatever size you need
# n_samples = 2000
# images_mat = torch.from_numpy(np.array(img_collection)[:n_samples]).float() / 255.0
# ground_truth_mask = create_circle_mask(img_size, radius=0.3)
# results = run_linear_regression_experiment(images_mat, ground_truth_mask, noise_level=5)
# visualize_results(results)

def run_regressors_Xy(images_mat, y_noised, test_images_mat, test_y_ground_truth, regressors=None, img_size=100):
    """Run linear regression experiment with different models."""
    if regressors is None:
        regressors = {
            "OLS": LinearRegression(),
            "Ridge": RidgeCV(alphas=list(np.logspace(-4, 5, 10))),
            "Lasso": LassoCV(alphas=list(np.logspace(-4, 5, 10))),
        }
    n_samples = images_mat.shape[0]
    images_mat = images_mat.reshape(n_samples, -1)
    results = {}
    for regressor_name, regressor in regressors.items():
        t0 = time()
        regressor.fit(images_mat, y_noised)
        t1 = time()
        
        coef_est = regressor.coef_
        r2_train = regressor.score(images_mat, y_noised)
        r2_test = regressor.score(test_images_mat, test_y_ground_truth)
        print(f"{regressor_name} R2: {r2_train:.3f}, {r2_test:.3f} | time: {t1-t0:.2f} seconds")
        results[f"coef_{regressor_name}"] = coef_est.reshape(img_size, img_size)
        results[f"r2_train_{regressor_name}"] = r2_train
        results[f"r2_test_{regressor_name}"] = r2_test
        results[f"time_{regressor_name}"] = t1-t0
    
    return results


class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = [join(root_dir, f"{i:05d}.png") for i in range(70000)]
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = img.resize((100, 100)).convert('L')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor

# Create dataset and dataloader
dataset = FFHQDataset(ffhq256)
dataloader = DataLoader(dataset, batch_size=100, num_workers=8, shuffle=False)

# Load all images
img_collection = []
for batch in tqdm(dataloader):
    img_collection.append(batch)
img_collection = torch.cat(img_collection, dim=0).unsqueeze(1)  # Add channel dimension

img_collection[0]  # Return first image

#%%
exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/AdvExampleLinearRegr/circ_mask_weights"
img_size = 100  # or whatever size you need
n_samples = img_collection.shape[0]
# img_mat_all = torch.from_numpy(np.array(img_collection)).float() / 255.0
img_mat_all = img_collection.squeeze(1)
img_mat_all = img_mat_all.view(img_mat_all.shape[0], -1)
ground_truth_mask = create_circle_mask(img_size, radius=0.3)
y_ground_truth = img_mat_all @ ground_truth_mask.flatten()
test_idx = np.arange(n_samples-1000, n_samples)
test_images_mat = img_mat_all[test_idx]
test_y_ground_truth = y_ground_truth[test_idx]
test_images_mat_np = test_images_mat.cpu().numpy()
test_y_ground_truth_np = test_y_ground_truth.cpu().numpy()
regressors = {
    "OLS": LinearRegression(),
    "Ridge": RidgeCV(alphas=list(np.logspace(-4, 5, 10))),
    "Lasso": LassoCV(alphas=list(np.logspace(-4, 5, 10)), n_jobs=-1, max_iter=5000),
}
for n_samples in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 65000,]:
    train_images_mat = img_mat_all[:n_samples]
    train_y_ground_truth = y_ground_truth[:n_samples]
    train_images_mat_np = train_images_mat.cpu().numpy()
    train_y_ground_truth_np = train_y_ground_truth.cpu().numpy()
    for noise_level in [0.000, 0.01, 0.1, 1.0, 3.0, 10.0,]:
        print(f"Running n_samples={n_samples}, noise_level={noise_level}")
        train_y_observed_np = train_y_ground_truth_np + np.random.randn(n_samples) * noise_level
        results = run_regressors_Xy(train_images_mat_np, train_y_observed_np, test_images_mat_np, test_y_ground_truth_np, regressors=regressors, img_size=img_size)
        results["ground_truth"] = ground_truth_mask
        results["n_samples"] = n_samples
        results["noise_level"] = noise_level
        results["NSR"] = noise_level**2/90**2
        pkl.dump(results, open(join(exproot, f"regression_results_n_samp{n_samples}_noise{noise_level}.pkl"), "wb"))
        # results = run_linear_regression_experiment(train_images_mat, train_y_observed, noise_level=noise_level, run_lasso=True)
        figh = visualize_results(results, title=f"n_samples={n_samples}, noise_level={noise_level}, NSR={noise_level**2/90**2:.2e}")
        saveallforms(exproot, f"regression_results_n_samp{n_samples}_noise{noise_level}", figh=figh)
        plt.close(figh)
    # raise Exception("Stop here")

