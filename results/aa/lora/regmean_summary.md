## knots_release

uniform best 52.3 (alpha 0.8); uniform alpha 1.3: 50.0

| budget | solve | regulariser | best dev acc (alpha) | acc at alpha 1.3 | median kappa(G) | median kappa(solve) |
|---|---|---|---|---|---|---|
| B0 | full | shrink 1.0 | 52.3 (0.8) | 50.0 | 9.85e+04 | 9.85e+04 |
| B0 | full | kappa<= 1e+02 ridge->mean | 52.3 (0.8) | 50.0 | 9.85e+04 | 1.00e+02 |
| B0 | full | kappa<= 1e+02 ridge->zero | 52.4 (1.0) | 52.4 | 9.85e+04 | 1.00e+02 |
| B0 | full | kappa<= 1e+03 ridge->mean | 52.3 (0.8) | 50.0 | 9.85e+04 | 1.00e+03 |
| B0 | full | kappa<= 1e+03 ridge->zero | 52.2 (0.8) | 51.0 | 9.85e+04 | 1.00e+03 |
| B0 | full | kappa<= 1e+04 ridge->mean | 52.3 (0.8) | 50.0 | 9.85e+04 | 1.00e+04 |
| B0 | full | kappa<= 1e+04 ridge->zero | 52.2 (0.8) | 50.1 | 9.85e+04 | 1.00e+04 |
| B0 | full | shrink 0.95 | 52.3 (0.8) | 50.0 | 2.53e+03 | 2.53e+03 |
| B0 | full | shrink 0.9 | 52.3 (0.8) | 50.0 | 1.54e+03 | 1.54e+03 |
| B0 | full | shrink 0.5 | 52.3 (0.8) | 50.0 | 3.90e+02 | 3.90e+02 |
| B0 | sub | shrink 1.0 | 52.3 (0.8) | 50.0 | 1.99e+02 | 1.99e+02 |
| B0 | sub | kappa<= 1e+02 ridge->mean | 52.3 (0.8) | 50.0 | 1.99e+02 | 1.00e+02 |
| B0 | sub | kappa<= 1e+02 ridge->zero | 52.3 (0.8) | 51.1 | 1.99e+02 | 1.00e+02 |
| B0 | sub | kappa<= 1e+03 ridge->mean | 52.3 (0.8) | 50.0 | 1.99e+02 | 1.99e+02 |
| B0 | sub | kappa<= 1e+03 ridge->zero | 52.3 (0.8) | 50.0 | 1.99e+02 | 1.99e+02 |
| B0 | sub | kappa<= 1e+04 ridge->mean | 52.3 (0.8) | 50.0 | 1.99e+02 | 1.99e+02 |
| B0 | sub | kappa<= 1e+04 ridge->zero | 52.3 (0.8) | 50.0 | 1.99e+02 | 1.99e+02 |
| B0 | sub | shrink 0.95 | 52.3 (0.8) | 50.0 | 1.69e+02 | 1.69e+02 |
| B0 | sub | shrink 0.9 | 52.3 (0.8) | 50.0 | 1.50e+02 | 1.50e+02 |
| B0 | sub | shrink 0.5 | 52.3 (0.8) | 50.0 | 5.44e+01 | 5.44e+01 |
| Bp | full | shrink 1.0 | 51.8 (0.8) | 38.5 | 8.68e+04 | 8.68e+04 |
| Bp | full | kappa<= 1e+02 ridge->mean | 51.4 (0.8) | 41.3 | 8.68e+04 | 1.00e+02 |
| Bp | full | kappa<= 1e+02 ridge->zero | 51.6 (0.8) | 47.8 | 8.68e+04 | 1.00e+02 |
| Bp | full | kappa<= 1e+03 ridge->mean | 51.6 (0.8) | 38.8 | 8.68e+04 | 1.00e+03 |
| Bp | full | kappa<= 1e+03 ridge->zero | 51.6 (0.8) | 40.3 | 8.68e+04 | 1.00e+03 |
| Bp | full | kappa<= 1e+04 ridge->mean | 51.8 (0.8) | 38.5 | 8.68e+04 | 1.00e+04 |
| Bp | full | kappa<= 1e+04 ridge->zero | 51.8 (0.8) | 38.7 | 8.68e+04 | 1.00e+04 |
| Bp | full | shrink 0.95 | 51.7 (0.8) | 38.3 | 3.03e+03 | 3.03e+03 |
| Bp | full | shrink 0.9 | 51.7 (0.8) | 38.4 | 1.64e+03 | 1.64e+03 |
| Bp | full | shrink 0.5 | 51.5 (0.8) | 39.9 | 4.07e+02 | 4.07e+02 |
| Bp | sub | shrink 1.0 | 51.1 (0.8) | 38.1 | 2.03e+02 | 2.03e+02 |
| Bp | sub | kappa<= 1e+02 ridge->mean | 50.8 (0.8) | 38.3 | 2.03e+02 | 1.00e+02 |
| Bp | sub | kappa<= 1e+02 ridge->zero | 51.1 (0.8) | 40.5 | 2.03e+02 | 1.00e+02 |
| Bp | sub | kappa<= 1e+03 ridge->mean | 51.1 (0.8) | 38.0 | 2.03e+02 | 2.03e+02 |
| Bp | sub | kappa<= 1e+03 ridge->zero | 51.1 (0.8) | 38.0 | 2.03e+02 | 2.03e+02 |
| Bp | sub | kappa<= 1e+04 ridge->mean | 51.1 (0.8) | 38.1 | 2.03e+02 | 2.03e+02 |
| Bp | sub | kappa<= 1e+04 ridge->zero | 51.1 (0.8) | 38.1 | 2.03e+02 | 2.03e+02 |
| Bp | sub | shrink 0.95 | 51.0 (0.8) | 38.1 | 1.74e+02 | 1.74e+02 |
| Bp | sub | shrink 0.9 | 51.0 (0.8) | 38.3 | 1.49e+02 | 1.49e+02 |
| Bp | sub | shrink 0.5 | 50.9 (0.8) | 39.2 | 5.59e+01 | 5.59e+01 |
| C | full | shrink 1.0 | 65.3 (0.8) | 4.1 | 1.05e+05 | 1.05e+05 |
| C | full | kappa<= 1e+02 ridge->mean | 60.5 (0.8) | 43.0 | 1.05e+05 | 1.00e+02 |
| C | full | kappa<= 1e+02 ridge->zero | 61.6 (1.0) | 53.9 | 1.05e+05 | 1.00e+02 |
| C | full | kappa<= 1e+03 ridge->mean | 63.9 (0.8) | 48.0 | 1.05e+05 | 1.00e+03 |
| C | full | kappa<= 1e+03 ridge->zero | 64.6 (0.8) | 49.7 | 1.05e+05 | 1.00e+03 |
| C | full | kappa<= 1e+04 ridge->mean | 65.5 (0.8) | 50.8 | 1.05e+05 | 1.00e+04 |
| C | full | kappa<= 1e+04 ridge->zero | 65.5 (0.8) | 50.9 | 1.05e+05 | 1.00e+04 |
| C | full | shrink 0.95 | 64.8 (0.8) | 50.5 | 3.35e+03 | 3.35e+03 |
| C | full | shrink 0.9 | 64.3 (0.8) | 50.2 | 1.97e+03 | 1.97e+03 |
| C | full | shrink 0.5 | 60.8 (0.8) | 43.3 | 4.24e+02 | 4.24e+02 |
| C | sub | shrink 1.0 | 57.9 (0.8) | 34.4 | 2.18e+02 | 2.18e+02 |
| C | sub | kappa<= 1e+02 ridge->mean | 56.1 (0.8) | 33.4 | 2.18e+02 | 1.00e+02 |
| C | sub | kappa<= 1e+02 ridge->zero | 57.7 (0.8) | 37.3 | 2.18e+02 | 1.00e+02 |
| C | sub | kappa<= 1e+03 ridge->mean | 57.9 (0.8) | 34.4 | 2.18e+02 | 2.18e+02 |
| C | sub | kappa<= 1e+03 ridge->zero | 57.9 (0.8) | 34.4 | 2.18e+02 | 2.18e+02 |
| C | sub | kappa<= 1e+04 ridge->mean | 57.9 (0.8) | 34.4 | 2.18e+02 | 2.18e+02 |
| C | sub | kappa<= 1e+04 ridge->zero | 57.9 (0.8) | 34.4 | 2.18e+02 | 2.18e+02 |
| C | sub | shrink 0.95 | 57.8 (0.8) | 34.5 | 1.84e+02 | 1.84e+02 |
| C | sub | shrink 0.9 | 57.4 (0.8) | 34.5 | 1.57e+02 | 1.57e+02 |
| C | sub | shrink 0.5 | 55.6 (0.8) | 35.7 | 5.88e+01 | 5.88e+01 |

- per-map kappa(G) C/full: median 1.05e+05, min 2.88e+03, max inf
- per-map kappa(G) C/sub: median 2.18e+02, min 1.13e+02, max 2.02e+03
- per-map kappa(G) Bp/full: median 8.68e+04, min 1.27e+03, max inf
- per-map kappa(G) Bp/sub: median 2.03e+02, min 9.35e+01, max 1.94e+03
- per-map kappa(G) B0/full: median 9.85e+04, min 1.03e+03, max inf
- per-map kappa(G) B0/sub: median 1.99e+02, min 7.89e+01, max 2.13e+03

## knots_release_n16

uniform best 52.3 (alpha 0.8); uniform alpha 1.3: 50.0

| budget | solve | regulariser | best dev acc (alpha) | acc at alpha 1.3 | median kappa(G) | median kappa(solve) |
|---|---|---|---|---|---|---|
| Bp | full | shrink 1.0 | 51.8 (0.8) | 38.5 | 1.19e+05 | 1.19e+05 |
| Bp | full | kappa<= 1e+02 ridge->mean | 51.1 (0.8) | 41.4 | 1.19e+05 | 1.00e+02 |
| Bp | full | kappa<= 1e+02 ridge->zero | 51.4 (0.8) | 47.6 | 1.19e+05 | 1.00e+02 |
| Bp | full | kappa<= 1e+03 ridge->mean | 51.2 (0.8) | 38.8 | 1.19e+05 | 1.00e+03 |
| Bp | full | kappa<= 1e+03 ridge->zero | 51.1 (0.8) | 40.3 | 1.19e+05 | 1.00e+03 |
| Bp | full | kappa<= 1e+04 ridge->mean | 51.5 (0.8) | 38.3 | 1.19e+05 | 1.00e+04 |
| Bp | full | kappa<= 1e+04 ridge->zero | 51.5 (0.8) | 38.4 | 1.19e+05 | 1.00e+04 |
| Bp | full | shrink 0.95 | 51.5 (0.8) | 39.0 | 3.54e+03 | 3.54e+03 |
| Bp | full | shrink 0.9 | 51.3 (0.8) | 39.0 | 1.99e+03 | 1.99e+03 |
| Bp | full | shrink 0.5 | 51.2 (0.8) | 40.3 | 4.48e+02 | 4.48e+02 |
| Bp | sub | shrink 1.0 | 51.2 (0.8) | 38.6 | 2.48e+02 | 2.48e+02 |
| Bp | sub | kappa<= 1e+02 ridge->mean | 51.1 (0.8) | 39.1 | 2.48e+02 | 1.00e+02 |
| Bp | sub | kappa<= 1e+02 ridge->zero | 51.2 (0.8) | 41.9 | 2.48e+02 | 1.00e+02 |
| Bp | sub | kappa<= 1e+03 ridge->mean | 51.2 (0.8) | 38.6 | 2.48e+02 | 2.48e+02 |
| Bp | sub | kappa<= 1e+03 ridge->zero | 51.2 (0.8) | 38.6 | 2.48e+02 | 2.48e+02 |
| Bp | sub | kappa<= 1e+04 ridge->mean | 51.2 (0.8) | 38.6 | 2.48e+02 | 2.48e+02 |
| Bp | sub | kappa<= 1e+04 ridge->zero | 51.2 (0.8) | 38.6 | 2.48e+02 | 2.48e+02 |
| Bp | sub | shrink 0.95 | 51.3 (0.8) | 38.8 | 2.04e+02 | 2.04e+02 |
| Bp | sub | shrink 0.9 | 51.2 (0.8) | 39.0 | 1.71e+02 | 1.71e+02 |
| Bp | sub | shrink 0.5 | 50.9 (0.8) | 39.6 | 6.01e+01 | 6.01e+01 |
| C | full | shrink 1.0 | 4.1 (0.8) | 4.1 | 2.19e+05 | 2.19e+05 |
| C | full | kappa<= 1e+02 ridge->mean | 58.1 (0.8) | 42.2 | 2.19e+05 | 1.00e+02 |
| C | full | kappa<= 1e+02 ridge->zero | 58.7 (0.8) | 51.4 | 2.19e+05 | 1.00e+02 |
| C | full | kappa<= 1e+03 ridge->mean | 60.5 (0.8) | 45.3 | 2.19e+05 | 1.00e+03 |
| C | full | kappa<= 1e+03 ridge->zero | 60.7 (0.8) | 46.7 | 2.19e+05 | 1.00e+03 |
| C | full | kappa<= 1e+04 ridge->mean | 62.1 (0.8) | 47.9 | 2.19e+05 | 1.00e+04 |
| C | full | kappa<= 1e+04 ridge->zero | 62.2 (0.8) | 48.0 | 2.19e+05 | 1.00e+04 |
| C | full | shrink 0.95 | 61.7 (0.8) | 47.7 | 4.26e+03 | 4.26e+03 |
| C | full | shrink 0.9 | 60.9 (0.8) | 47.4 | 2.33e+03 | 2.33e+03 |
| C | full | shrink 0.5 | 58.1 (0.8) | 42.3 | 4.50e+02 | 4.50e+02 |
| C | sub | shrink 1.0 | 57.3 (0.8) | 36.3 | 2.84e+02 | 2.84e+02 |
| C | sub | kappa<= 1e+02 ridge->mean | 55.0 (0.8) | 33.1 | 2.84e+02 | 1.00e+02 |
| C | sub | kappa<= 1e+02 ridge->zero | 56.4 (0.8) | 37.7 | 2.84e+02 | 1.00e+02 |
| C | sub | kappa<= 1e+03 ridge->mean | 56.8 (0.8) | 34.8 | 2.84e+02 | 2.84e+02 |
| C | sub | kappa<= 1e+03 ridge->zero | 56.8 (0.8) | 34.9 | 2.84e+02 | 2.84e+02 |
| C | sub | kappa<= 1e+04 ridge->mean | 57.3 (0.8) | 36.3 | 2.84e+02 | 2.84e+02 |
| C | sub | kappa<= 1e+04 ridge->zero | 57.3 (0.8) | 36.3 | 2.84e+02 | 2.84e+02 |
| C | sub | shrink 0.95 | 57.0 (0.8) | 35.4 | 2.30e+02 | 2.30e+02 |
| C | sub | shrink 0.9 | 56.5 (0.8) | 34.8 | 1.94e+02 | 1.94e+02 |
| C | sub | shrink 0.5 | 54.7 (0.8) | 35.0 | 6.66e+01 | 6.66e+01 |

- per-map kappa(G) C/full: median 2.19e+05, min 1.90e+04, max inf
- per-map kappa(G) C/sub: median 2.84e+02, min 1.34e+02, max 4.36e+03
- per-map kappa(G) Bp/full: median 1.19e+05, min 4.96e+03, max inf
- per-map kappa(G) Bp/sub: median 2.48e+02, min 1.16e+02, max 1.83e+03

## knots_release_n64

uniform best 52.3 (alpha 0.8); uniform alpha 1.3: 50.0

| budget | solve | regulariser | best dev acc (alpha) | acc at alpha 1.3 | median kappa(G) | median kappa(solve) |
|---|---|---|---|---|---|---|
| C | full | shrink 1.0 | 4.1 (0.8) | 4.1 | 1.13e+05 | 1.13e+05 |
| C | full | shrink 0.95 | 63.9 (0.8) | 49.9 | 3.63e+03 | 3.63e+03 |
| C | full | shrink 0.9 | 63.3 (0.8) | 49.5 | 2.06e+03 | 2.06e+03 |
| C | full | shrink 0.5 | 59.9 (0.8) | nan | 4.30e+02 | 4.30e+02 |

