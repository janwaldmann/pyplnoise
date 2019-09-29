# pyplnoise
**Py**thon **p**ower **l**aw **noise** generates arbitrarily long streams of power law noise
using NumPy and SciPy.

The noise is generated with shaping filters and the methodology closely follows
[Stephane Plaszczynski, Fluct. Noise Lett. 7: R&ndash;R13, 2007](https://doi.org/10.1142/S0219477507003635).
You can also find the article on [arXiv:astro-ph/0510081](https://arxiv.org/abs/astro-ph/0510081).

**pyplnoise** consists of a single module providing classes implementing the following
noise sources:
* 1/f<sup>&alpha;</sup> power law noise with upper and lower frequency limits (class `AlphaNoise`),
* a fast red (Brownian) noise generator with a lower frequency limit (class `RedNoise`),
* and of course white noise (class `WhiteNoise`).

## Quick Example
The interface is very simple: just instantiate one of the above classes and run
`get_sample()` to retrieve a single sample or `get_series(npts)` to
retrieve an array of `npts` samples. Interface documentation is available in the code.

```python
import pyplnoise
import numpy as np

fs = 10. # sampling frequency in Hz

# instantiate a noise source with lower frequency limit 1e-3 Hz,
# upper frequency limit 5 Hz and 1/f power spectrum
noisegen = pyplnoise.AlphaNoise(fs, 1e-3, fs/2., alpha=1.0)

one_sample = noisegen.get_sample()
many_samples = noisegen.get_series(1000000)
```

## Detailed Examples
Jupyter notebooks are provided in the [/examples](/examples) directory:
1. [Overview of the noise sources and their properties](/examples/overview_of_noise_sources.ipynb)
2. [Application example: modeling the random signal errors of a gyroscope (Allan variance
   of synthetic noise)](/examples/application_example_allan_variance.ipynb)

## Installation and dependencies
### Dependencies
* NumPy &ge; 1.17 (see NEP 19)
* SciPy &ge; 1.3

### Installing directly from GitHub
Download the release tarball and run
```python
python setup.py install
```

Because everything is contained in the module `pyplnoise`, you can alternatively just copy
the module and the LICENSE file into your project.

## You may find pyplnoise useful, if...
...you're looking to generate 1/f<sup>&alpha;</sup> noise with very long correlation
times (frequencies &ll; 10<sup>-7</sup> Hz); particularly if your machine has very limited
memory resources.

## You may *not* find pyplnoise useful, if...
* ...you're looking for a pink noise source for your software synthesizer or other audio stuff.
  There are lots of interesting solutions for such applications, notably
  "[A New Shade of Pink](https://github.com/Stenzel/newshadeofpink)",
  the [Voss-McCartney Algorithm, which is also available in Python](https://www.dsprelated.com/showarticle/908.php)
  and [some highly specialized filters](http://www.firstpr.com.au/dsp/pink-noise/).
* ...you want to generate finite 1/f<sup>&alpha;</sup> noise streams with relatively short
  correlation times (frequencies &ge; 10<sup>-7</sup> Hz). In such a case [Fourier transform
  methods](https://github.com/felixpatzelt/colorednoise) are tractable and these methods deliver
  higher quality results than the shaping filters used by **pyplnoise**.

