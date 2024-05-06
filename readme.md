# Introduction

## SAM License

SAM is licensed under the Apache 2.0 license [Apache 2.0 license](SAM_LICENSE)

## SegGPT License

SegGPT is licensed under the MIT license [MIT license](SegGPT_LICENSE)

# Install from source
```
git clone git@github.com:MetroStar/labelup-core.git

pip install -e labelup-core

```

# Build the distribution


```
python -m pip install --upgrade build

python -m build
```
and install from wheel

```
pip install dist/labelup-0.1-py3-none-any.whl
```

# Or install via conda or mamba
Update environment.yml with `pytorch-cuda` compatible with your cuda version (pytorch.org)
```
git clone git@github.com:MetroStar/labelup-core.git

cd labelup-core

mamba env create -n labelup-core -f environment.yml

mamba activate labelup-core

pip install -e . --no-deps
