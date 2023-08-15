def get_default_config():
    return dict(
        Prediction=dict(
            arch1=[48, 96, 48],
            arch2=[48, 96, 48],
        ),
        Autoencoder=dict(
            arch1=[244, 128, 128, 48],
            arch2=[270, 128, 128, 48],
            activations1='relu',
            activations2='relu',
            batchnorm=True,
        ),
        training=dict(
            seed=42,
            start_dual_prediction=50,
            epoch=1000,
            lr=0.01,
            # Balanced factors for L_intra_cl, L_intra_rec
            alpha=9,
            lambda1=1,  # intra loss
            lambda2=1,  # rec loss
            sigma=0.0001,  # poi view
        ),
    )

