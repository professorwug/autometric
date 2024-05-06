# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/jax-geodesics.ipynb.

# %% auto 0
__all__ = ['DummyOracle', 'wrap_torch_metric', 'GeodesicQuicktrainer', 'sample_along_geodesic', 'plot_3d_with_geodesics',
           'visualize_geodesics']

# %% ../../nbs/library/jax-geodesics.ipynb 20
class DummyOracle:
    def __init__(self):
        return
    
    def geo_length(self,x0,x1):
        """
        Computes the length between x0 and x1 on the sphere
        """
        return jnp.ones(len(x0))
    
    def mse_geodesic(self, x0,x1,t, preds):
        return 1

# %% ../../nbs/library/jax-geodesics.ipynb 21
def wrap_torch_metric(x, metric):
    print(x)
    print(x.shape)
    x_np = np.array(x)
    x = torch.from_numpy(x)
    return jax.lax.stop_gradient(metric(x).detach().numpy())

class GeodesicQuicktrainer(GeodesicTrainer):
    def __init__(self, 
                 X:np.ndarray, # data coordinates, in ambient dimension
                 intrinsic_dim:int, # dimension (usually intrinsic dimension of manifold)
                 metric_fn, # a function that, given (n, d) input, returns the metric for each of n points.
                 max_epochs=1000,
                 batch_size = 32,
                 seed = 42,
                 layers_in_curve = 3,
                 hidden_dimension=16,
                 use_autometric_metric = True,
                 **kwargs
                 ):
        #split X in two
        X = np.array(X)
        np.random.seed(seed)
        idx = np.random.permutation(len(X))
        X = X[idx]
        X1 = X[:len(X)//2]
        X2 = X[len(X)//2:]
        datamodule = GeodesicDataModule(X1, X2, batch_size = batch_size)
        cond_curve = CondCurve(
            input_dim = intrinsic_dim,
            hidden_dim = hidden_dimension,
            scale_factor = 5,
            symmetric = False,
            num_layers=layers_in_curve,
        )
        oracle = DummyOracle()
        if use_autometric_metric:
            metric_fn = partial(wrap_torch_metric, metric = metric_fn)

        super().__init__(
            datamodule, # a class with functions train_dataloader, val_dataloader, test_dataloader which return those. Each batch should return two points, x_o, and x_1
            cond_curve, # the curve to be learned
            metric_fn, # a function that, given (n, d) input, returns the metric for each of n points.
            oracle, # 
            max_epochs = max_epochs,
            lr = 0.001,
            density_lambda=0,
            seed = 421,
            log_every_n_steps=10,
            n_interp_times=20,
            k_density=5, # default parameters in EB's hydra configs
            logger = False,
            callbacks = [],
            )
    

# %% ../../nbs/library/jax-geodesics.ipynb 49
from .utils import plot_3d

def sample_along_geodesic(
    start, end, 
    geodesic_func, encoder, decoder,
    num_times = 50,
):
    start_latent = encoder(start).detach().cpu().numpy()[0]
    end_latent = encoder(end).detach().cpu().numpy()[0]
    ts = jnp.linspace(0,1,num_times)[:,None]
    # partial_geodesic = partial(geodesic_func, x0=start_latent, x1=end_latent)
    # print(geodesic_func(start_latent, end_latent, 0.5))
    samples = geodesic_func(start_latent, end_latent, ts)
    samples = np.squeeze(np.array(jax.device_get(samples))) # convert back to numpy
    samples_decoded = decoder(samples)
    return samples_decoded

def plot_3d_with_geodesics(X, geodesics):
    # if geodesics is not a list, wrap it in one
    if isinstance(geodesics, np.ndarray):
        geodesics = [geodesics]
    combined_geodesics = np.concatenate(geodesics, axis=0)
    all_points = np.concatenate([X, combined_geodesics], axis=0)
    plot_colors = np.zeros(len(X) + len(combined_geodesics))
    running_length = len(X)
    for i, g in enumerate(geodesics):
        plot_colors[running_length:running_length + len(g)] = i + 1
        running_length += len(g)
    plot_3d(all_points, plot_colors, use_plotly=True)

def visualize_geodesics(
    X_ambient:np.ndarray, # ambient coordinates
    geodesic_func_1, # first geodesic function. Takes input x1, x2, t and returns the geodesic at time t.
    geodesic_func_2, 
    model1,
    model2,
    num_geodesics_to_sample:int = 1,
):
    encoder_1 = model1.encode
    decoder_1 = model1.decode
    encoder_2 = model2.encode
    decoder_2 = model2.decode
    endpoints_idx = np.random.choice(np.arange(len(X_ambient)), size=(num_geodesics_to_sample,2), replace=False)
    geodesics_1 = []
    geodesics_2 = []
    for i in range(num_geodesics_to_sample):
        start = np.array(X_ambient[endpoints_idx[i][0]][None,:])
        end = np.array(X_ambient[endpoints_idx[i][1]][None,:])
        geodesics_1.append(
            sample_along_geodesic(start, end, geodesic_func_1, encoder_1, decoder_1)
        )
        geodesics_2.append(
            sample_along_geodesic(start, end, geodesic_func_2, encoder_2, decoder_2)
        )
        
    combined_g1s = np.concatenate(geodesics_1, axis=0)
    combined_g2s = np.concatenate(geodesics_2, axis=0)
    plot_3d_with_geodesics(X_ambient, [combined_g1s, combined_g2s])
    return geodesics_1, geodesics_2


