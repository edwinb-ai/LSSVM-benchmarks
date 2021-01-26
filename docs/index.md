# SVM Benchmarks

In this directory we will be compiling the results as HTML files.

## Baseline

1. [Multiclass classification. Small dataset.](./baseline_classification.jl.html)

## Source files

The original files are done using the [Pluo.jl](https://github.com/fonsp/Pluto.jl)
development environment.
They can be run locally if you clone this repository on your machine using

```shell
git clone https://github.com/edwinb-ai/LSSVM-benchmarks.git
```

and then, in the `Julia` REPL

```julia
julia> using Pkg
julia> Pkg.activate()
julia> Pkg.instantiate()
```

This will activate the enviroment and download all the dependencies.
After that you might want to start a _Pluto_ server like so

```julia
julia> using Pluto
julia> Pluto.run()
```

and after that you can run the notebooks, which can be found in the `src` directory
within this repository.
