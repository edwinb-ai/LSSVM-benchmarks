### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 01db1fc8-5ffd-11eb-0dae-cda0ff89e63d
begin
	using MLJ, MLJModels
	using Random
	using BenchmarkTools
	using DataFrames
	using LeastSquaresSVM
end

# ╔═╡ b84726f0-6003-11eb-2e0d-6f14682c9588
md"""
# Baseline benchmark for multiclass classification

In this document we will benchmark the multiclass classification problem of the
Least Squares Support Vector Classifier from `LeastSquaresSVM` and the one from
the fantastic and very well-known `libSVM` solver.

The idea is to test both their prediction and training complexities, both
in time and space. To accomplish this, we will be using the `BenchmarkTools`
package.

We will test **two** cases here:

1. Training _with_ hyperparameter tuning;
2. training _without_ tuning.

This is because a well-tuned SVM is _very_ important to predict good results.
We almost always need a well-tuned SVM, so this is an important step.

The dataset we'll be using is the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris/).
This is a very simple and easy dataset, and we will deal with larger datasets
in another document elsewhere.
"""

# ╔═╡ 5d73b9a0-6005-11eb-2ee6-2164f4169fea
md"""
First, some imports and all that good stuff.
"""

# ╔═╡ 143d4bd4-5ffe-11eb-23dc-1751a1913a05
@load SVC pkg=LIBSVM

# ╔═╡ 6a111cfe-6005-11eb-15df-875c96454e44
md"""
## Loading the dataset

We now load the dataset, and we'll convert it to a `DataFrame` to inspect it.
We just want to check that the `scitypes` are correct.
"""

# ╔═╡ 281afffa-5ffd-11eb-24ce-219f746515da
X, y = @load_iris;

# ╔═╡ 6e9dcf16-5ffd-11eb-0523-e1cfd36fd57a
dfIris = DataFrame(X);

# ╔═╡ 82244ce8-5ffd-11eb-003d-23214d962d20
dfIris.y = y;

# ╔═╡ 878df3be-5ffd-11eb-2b19-7f74692d0189
first(dfIris, 3)

# ╔═╡ 8c358782-6005-11eb-2787-8b32b5749b57
md"""
## Training setup

Great, so everything looks good with the dataset.
Let us now setup the dataset for training and testing. We'll do a 60-40% split
on this dataset.
Recall that this dataset only has 150 datasamples. We also always shuffle the
splitting.
"""

# ╔═╡ 2bbc10ce-6000-11eb-2162-398983b5cf48
train, test = partition(eachindex(y), 0.6, shuffle=true);

# ╔═╡ cb51aff4-6005-11eb-1a7e-6b229f087a31
md"""
## Training functions

To correctly benchmark the training steps, we need to setup and create some
functions that collect all the training logic.

Here, we are doing two pairs of functions, one _without_ hyperparameter tuning
and the other _with_ tuning.
You can tell them apart because the ones _with_ hyperparameter tuning have
the suffix `hp`.

Also, to differentiate between both implementations, `svm` corresponds to the
`libsvm` implementation. On the other hand, `lssvm` corresponds to the
`LeastSquaresSVM` implementation.
"""

# ╔═╡ f6f1ffe8-6004-11eb-0d26-3d51bcb24d61
function train_lssvm(X, y, train)
	model = LSSVClassifier()
	
	mach = machine(model, X, y)
	fit!(mach, rows=train, verbosity=0)
	
	return mach
end;

# ╔═╡ feaf46c8-6004-11eb-1fe3-7792a33bbc0e
function train_svm(X, y, train)
	model = SVC()
	
	mach = machine(model, X, y)
	fit!(mach, rows=train, verbosity=0)
	
	return mach
end;

# ╔═╡ 92b48212-5ffd-11eb-3032-d1c9519703f2
function train_lssvm_hp(X, y, train)
	model = LSSVClassifier()
	
	r1 = range(model, :σ, lower=1e-2, upper=10.0)
	r2 = range(model, :γ, lower=1, upper=200.0)
	
	self_tuning_model = TunedModel(
		model=model,
		tuning=Grid(goal=500),
		resampling=CV(nfolds=5),
		range=[r1, r2],
		measure=accuracy
	)
	
	mach = machine(self_tuning_model, X, y)
	fit!(mach, rows=train)
	
	return mach
end;

# ╔═╡ 5973126c-5ffe-11eb-2219-85731b9c0786
function train_svm_hp(X, y, train)
	model = SVC()
	
	r1 = range(model, :cost, lower=1, upper=1000.0)
	r2 = range(model, :gamma, lower=1e-2, upper=10.0)
	
	self_tuning_model = TunedModel(
		model=model,
		tuning=Grid(goal=500),
		resampling=CV(nfolds=5),
		range=[r1, r2],
		measure=accuracy
	)
	
	mach = machine(self_tuning_model, X, y)
	fit!(mach, rows=train)
	
	return mach
end;

# ╔═╡ 2ee014fc-6006-11eb-367a-dbc9f9f97225
md"""
## Dataset prep

With that out of the way, we need to remove the mean from the dataset, as
well as setting its standard deviation to one.
This is referred to as **standardizing** the dataset, and `MLJ` already has
a model for such tasks.

We'll standardize our model before feeding it to our models, because SVMs
are quite picky of non-standardized data.
"""

# ╔═╡ 5e76126c-5ffe-11eb-2b0b-6fe6136909c5
Xstd = MLJ.transform(MLJ.fit!(MLJ.machine(Standardizer(), X)), X);

# ╔═╡ 6f0d1bec-6006-11eb-106b-bd780f26fef9
md"""
## Benchmarking

Let's begin with benchmarking.

### Without tuning

First, let's do a baseline benchmark, _without_ hyperparameter tuning.
Here, we are looking for time and space estimations for both implementations.

We'll start with the `LeastSquaresSVM` implementation.
"""

# ╔═╡ 0ab3cf70-6005-11eb-3f80-03a804a70776
@benchmark train_lssvm($Xstd, $y, $train)

# ╔═╡ ad027e4a-6006-11eb-3ae1-f11f7d5fecdd
md"""
#### Memory allocation

Right out of the bat, we can see that there is quite a lot of allocation.
In the order of KiB it might not seem much, but this will surely be
more promiment in the hyperparameter tuning estimations.

This is expected due to the following reasons:

- This formulation will **always** need **all** the dataset for both training and testing.
- The solver (conjugate gradient, the iterative formulation) has to deal with allocations due to having to create a basis for each iteration.

#### Time estimation

Following the [manual for `BenchmarkTools`](https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/doc/manual.md#which-estimator-should-i-use) we will use the **median** as out metric for
comparing both models. This is because it is not really affected by
noise or some other type of outliers.
"""

# ╔═╡ 85b1dd2c-6008-11eb-0689-9572c7411b30
md"""
With this in mind our median time is of ≈ 400 microseconds, which is pretty okay.
Some of this might come from some compiler optimizations and clever machine
code.

We'll move on to the next implementation to have something to compare it with.
"""

# ╔═╡ 0a75a496-6005-11eb-2135-eb8b6d4d106f
@benchmark train_svm($Xstd, $y, $train)

# ╔═╡ 8ee94646-6008-11eb-089a-7b2ea0a9497b
md"""
#### Memory allocation

Wow, this is an order of maginitude lower compared to the `LeastSquaresSVM`
implementation. But then, again, this is expected.
The `libSVM` solver is **sparse**, meaning that it only selects a couple of data
instances as its _support vectors_.

#### Time estimation

By comparing with the previous _median_ value, this implementation is faster,
and better.
I kind of expected this result due to the fact the the `libSVM` solver is very
_highly_ optimized code, and written almost entirely in C/C++.

And by highly optimized code I mean that they use really clever tricks in various
numerical computations throughout their implementation.

But even so the `LeastSquaresSVM` implementation is very close. Not too shabby.
"""

# ╔═╡ f6c9278a-6009-11eb-300b-1fb5f3a9ad13
md"""
### With tuning. Setup and general info.

The hyperparameter search is done with the [`Grid`](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#MLJTuning.Grid)
algorithm from `MLJ`, which is a conventional grid seach.

The reason for this choice is simple: we want to be fair.

Each model tunes its hyperparameters using 5-fold cross-validation on the
_training_ set only.

Each model has to tune **two** hyperparameters:

- an _intrinsic_ one; _cost_ for the `libSVM` and _γ_ for the `LeastSquaresSVM`,
- the kernel hyperparameter; both models are using the same RBF kernel.

Each model will search for the best pair of hyperparameters in a grid of
$[75 \times 75]$ for a total of 1500 points in the grid.
"""

# ╔═╡ 5e3c2068-5ffe-11eb-3173-59eb4527939f
@benchmark train_lssvm_hp($Xstd, $y, $train) samples=3

# ╔═╡ 66054bac-6002-11eb-063c-5525eb2a0ca1
mach1 = train_lssvm_hp(Xstd, y, train);

# ╔═╡ 5e286ffa-5ffe-11eb-22c3-2bba58505b97
@benchmark train_svm_hp($Xstd, $y, $train)

# ╔═╡ 6e868db0-6002-11eb-0cc1-edf2da791ccb
mach2 = train_svm_hp(Xstd, y, train);

# ╔═╡ 5e0dc6e6-5ffe-11eb-0b25-8fa52bd3cfc1
function predict_model(y, test, mach)
	results = MLJ.predict(mach, rows=test)
	acc = MLJ.accuracy(results, y[test])
	
	return acc
end;

# ╔═╡ c55d8cb4-6001-11eb-2e6b-655ac8539748
@benchmark predict_model($y, $test, $mach1)

# ╔═╡ 949df2e8-6002-11eb-14cb-e5a269cf2846
acc1 = predict_model(y, test, mach1)

# ╔═╡ c5477f70-6001-11eb-3a08-87356dfbbd29
@benchmark predict_model($y, $test, $mach2)

# ╔═╡ c53166f2-6001-11eb-19d9-5d104a8e7107
acc2 = predict_model(y, test, mach2)

# ╔═╡ c4d94f5a-6001-11eb-2d11-5da129bacc2d


# ╔═╡ c4c13726-6001-11eb-277a-910dd8343dc1


# ╔═╡ 5df6ace0-5ffe-11eb-2e3c-d9239d9f8825


# ╔═╡ 5de00670-5ffe-11eb-1e3b-b5d9fa3fca77


# ╔═╡ 5dc91bfc-5ffe-11eb-34b9-b56bbda3dae3


# ╔═╡ 5db23042-5ffe-11eb-0674-6582ebdcebf7


# ╔═╡ 5d963f86-5ffe-11eb-2a1f-2b1fc3ee9e42


# ╔═╡ 5d7c9e6e-5ffe-11eb-2e96-1fce4d8fe9d4


# ╔═╡ 5d659dd6-5ffe-11eb-231b-1b067477d9f2


# ╔═╡ 5d4ba0b6-5ffe-11eb-1942-89b9c27cfd3e


# ╔═╡ 5d30d6fa-5ffe-11eb-07f0-9baa93e3e25c


# ╔═╡ 5d16ccae-5ffe-11eb-3f0c-43cb5281ce17


# ╔═╡ 5cf6eafa-5ffe-11eb-2f02-33e7b1f34f33


# ╔═╡ 5cd047fe-5ffe-11eb-2e11-abad1cb7ba45


# ╔═╡ 5c85908a-5ffe-11eb-1407-f311ce238968


# ╔═╡ Cell order:
# ╟─b84726f0-6003-11eb-2e0d-6f14682c9588
# ╟─5d73b9a0-6005-11eb-2ee6-2164f4169fea
# ╠═01db1fc8-5ffd-11eb-0dae-cda0ff89e63d
# ╠═143d4bd4-5ffe-11eb-23dc-1751a1913a05
# ╟─6a111cfe-6005-11eb-15df-875c96454e44
# ╠═281afffa-5ffd-11eb-24ce-219f746515da
# ╠═6e9dcf16-5ffd-11eb-0523-e1cfd36fd57a
# ╠═82244ce8-5ffd-11eb-003d-23214d962d20
# ╠═878df3be-5ffd-11eb-2b19-7f74692d0189
# ╟─8c358782-6005-11eb-2787-8b32b5749b57
# ╠═2bbc10ce-6000-11eb-2162-398983b5cf48
# ╟─cb51aff4-6005-11eb-1a7e-6b229f087a31
# ╠═f6f1ffe8-6004-11eb-0d26-3d51bcb24d61
# ╠═feaf46c8-6004-11eb-1fe3-7792a33bbc0e
# ╠═92b48212-5ffd-11eb-3032-d1c9519703f2
# ╠═5973126c-5ffe-11eb-2219-85731b9c0786
# ╟─2ee014fc-6006-11eb-367a-dbc9f9f97225
# ╠═5e76126c-5ffe-11eb-2b0b-6fe6136909c5
# ╟─6f0d1bec-6006-11eb-106b-bd780f26fef9
# ╠═0ab3cf70-6005-11eb-3f80-03a804a70776
# ╟─ad027e4a-6006-11eb-3ae1-f11f7d5fecdd
# ╟─85b1dd2c-6008-11eb-0689-9572c7411b30
# ╠═0a75a496-6005-11eb-2135-eb8b6d4d106f
# ╟─8ee94646-6008-11eb-089a-7b2ea0a9497b
# ╠═f6c9278a-6009-11eb-300b-1fb5f3a9ad13
# ╠═5e3c2068-5ffe-11eb-3173-59eb4527939f
# ╠═66054bac-6002-11eb-063c-5525eb2a0ca1
# ╠═5e286ffa-5ffe-11eb-22c3-2bba58505b97
# ╠═6e868db0-6002-11eb-0cc1-edf2da791ccb
# ╠═5e0dc6e6-5ffe-11eb-0b25-8fa52bd3cfc1
# ╠═c55d8cb4-6001-11eb-2e6b-655ac8539748
# ╠═949df2e8-6002-11eb-14cb-e5a269cf2846
# ╠═c5477f70-6001-11eb-3a08-87356dfbbd29
# ╠═c53166f2-6001-11eb-19d9-5d104a8e7107
# ╠═c4d94f5a-6001-11eb-2d11-5da129bacc2d
# ╠═c4c13726-6001-11eb-277a-910dd8343dc1
# ╠═5df6ace0-5ffe-11eb-2e3c-d9239d9f8825
# ╠═5de00670-5ffe-11eb-1e3b-b5d9fa3fca77
# ╠═5dc91bfc-5ffe-11eb-34b9-b56bbda3dae3
# ╠═5db23042-5ffe-11eb-0674-6582ebdcebf7
# ╠═5d963f86-5ffe-11eb-2a1f-2b1fc3ee9e42
# ╠═5d7c9e6e-5ffe-11eb-2e96-1fce4d8fe9d4
# ╠═5d659dd6-5ffe-11eb-231b-1b067477d9f2
# ╠═5d4ba0b6-5ffe-11eb-1942-89b9c27cfd3e
# ╠═5d30d6fa-5ffe-11eb-07f0-9baa93e3e25c
# ╠═5d16ccae-5ffe-11eb-3f0c-43cb5281ce17
# ╠═5cf6eafa-5ffe-11eb-2f02-33e7b1f34f33
# ╠═5cd047fe-5ffe-11eb-2e11-abad1cb7ba45
# ╠═5c85908a-5ffe-11eb-1407-f311ce238968
