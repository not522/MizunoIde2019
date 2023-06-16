# Usage

## Download velocity structure files

Run `download.sh` in the JMA2001 directory.

## For 1 day

```
python3 src/main.py --format example.in --date 100101
```

## For 1 time window

```
python3 src/main.py --format example.in --date 100101 --time_window 0
```

## For days, months, years

```
./run.sh
```

You can run in parallel. Change -j option of "parallel" command.

## Plot

```
python3 src/main.py --format example.in --date 100101 --time_window 0 --mode plot
```

# Specify the data for the calculation
Change format option.

# Requirements
## Python libraries
- NumPy
- SciPy
- ObsPy
- tqdm
- matplotlib, Basemap (optional; for plotting data)

## Linux commands
- parallel

# Changes from the paper
- Use the 50% confidence interval as the source location error instead of the standard deviation.
- Use "SLSQP" non-linear optimization algorithm instead of "CCSA".

# Reference
Mizuno, N., & Ide, S. (2019). Development of a modified envelope correlation method based on maximum-likelihood method and application to detecting and locating deep tectonic tremors in western Japan. Earth, Planets and Space, 71(1), 40.
