# Domains -- the method by application

The [experiment notebooks](Experiments) are organised by what they establish.
These pages read the same results by application: for each domain, what is
asked of an estimator there, which established methods it is scored against,
what the notebooks measure, and where dtfit loses. NLLS is the reference
throughout; where theory gives the image no advantage, parity is reported as
parity.

| Domain | dtfit routes | Scored against | Notebooks |
|---|---|---|---|
| [Parameter estimation](Domain-Parameter-Estimation) | `fit` in the Legendre and block bases, the robust image, `basis="auto"` | SciPy NLLS, robust NLLS, the NIST certified values | 15, 14, 31 |
| [Forecasting](Domain-Forecasting) | a declared structure fitted from the image, `auto_forecast` | random walk, seasonal naive, drift, ETS, Theta, (S)ARIMA, MLP, the published LTSF numbers | 32 |
| [Embedded control](Domain-Embedded-Control) | `ImageFilter` in both bases, the per-sample innovation test, `FilterBank` | a parameter-space EKF over a grid of settings, RLS, Kalman-CA, a sliding-window refit | 21, 24 |
| [Big data](Domain-Big-Data) | `Image.merge`, `ImageStream`, `fit_many`, the channel projection on the GPU | the resident fit, a serial loop, polynomial surrogates | 22 |
| [Real-time GPS](Domain-Realtime-GPS) | the streaming Legendre and block trackers, IMU regressors, the robust window | Kalman-CA, a coordinated-turn EKF, an IMM, a Huber-hardened Kalman | 33, 34 |
| [GPS hardware rig](Domain-Realtime-GPS-Hardware) | the on-chip float32 filter, the host trackers on recorded drives | Kalman-CA, CT-EKF on the same fixes | rig |
| [Stochastic series](Domain-Stochastic-Series) | the second-order image, the regime router, the merged forecaster | OLS AR(1), GARCH quasi-MLE, R/S, DFA, ETS, ARIMA, the random walk | 23, 36 |
| [Archive showcase](Domain-Image-Showcase) | `ImageStream`, fits from stored images, the image on the wire | `numpy.linalg.lstsq` on the raw rows, MIDAS velocities, NOAA normals | 35 |
