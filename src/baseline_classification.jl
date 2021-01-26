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
		tuning=Grid(goal=625),
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
		tuning=Grid(goal=625),
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

Each model will search for the best pair of hyperparameters in a 
square grid of 625 points. That is, 25 values for each hyperparameter.

It might seem like very few elements, but keep in mind that no parallelization
is used for this search (which normally it is recommended to do).
To make the comparison fair, we are doing this in a single thread, for each
model.

The performance metric will be the classification _accuracy._
"""

# ╔═╡ 5e3c2068-5ffe-11eb-3173-59eb4527939f
t1 = @benchmark train_lssvm_hp($Xstd, $y, $train)

# ╔═╡ da5d6972-602b-11eb-3cce-a50ed90928e0
md"""
#### Memory allocations

Well, this is not a surprise. For each pair of points, a model is trained.
The whole dataset set (the training portion of it) is used each time. For now
it is only comprised of $(length(train)) data instances. But imagine how large
can this grow for datasets with more than just 150 data samples.

This is one of the reasons why the least squares formulation is not the popular
one. Even though it is very simple to implement, it consumes memory like there
is no tomorrow.

#### Time estimation

This is quite a surprise. I seriously did not expect it to perform as well as
it does.
It is fast, efficient, but I think this is mostly due to the nature of the iterative
solver it uses.
If a more direct approach were used, I firmly believe the running time would have
blown up.
"""

# ╔═╡ 5e286ffa-5ffe-11eb-22c3-2bba58505b97
t2 = @benchmark train_svm_hp($Xstd, $y, $train)

# ╔═╡ abe9a28a-602c-11eb-18a9-493b4b05d461
md"""
#### Memory allocations

Again, this is not a surprise. By using just a few data instances, the `libSVM`
implementation is _very_ conservative, memory-wise.

Do note, however, that the difference in memory usage is mind-blowing.
The difference between this implementation and the least squares one is
$(round(t1.memory / t2.memory; digits=4)) smaller.

#### Time estimation

This is really the weird part. I would have expected that this implementation
would be faster.
Maybe it has something to do with the number of samples from the benchmark.
Anyway, the results show that both implementations are equivalent.

This is great news, because it means that no matter which implementation you use,
you will always have your results in a timely manner.
You can choose between both implementations and the prediction time is almost the
same.
"""

# ╔═╡ 00fdae46-601a-11eb-1db6-e73e0f4ead1e
md"""
## Prediction

In this last section I just wanted to show the prediction time for both
implementations.

We are assuming that both models have been tuned correctly.

First, I will wrap the prediction steps in a function, evaluation the
_accuracy_ of prediction for each model.
"""

# ╔═╡ 5e0dc6e6-5ffe-11eb-0b25-8fa52bd3cfc1
function predict_model(y, test, mach)
	results = MLJ.predict(mach, rows=test)
	acc = MLJ.accuracy(results, y[test])
	
	return acc
end;

# ╔═╡ 61c5201a-602e-11eb-15a7-1f0334dfa695
md"""
I need a tuned least squares SVM, so I do that first.
"""

# ╔═╡ 66054bac-6002-11eb-063c-5525eb2a0ca1
mach1 = train_lssvm_hp(Xstd, y, train);

# ╔═╡ c55d8cb4-6001-11eb-2e6b-655ac8539748
@benchmark predict_model($y, $test, $mach1)

# ╔═╡ 949df2e8-6002-11eb-14cb-e5a269cf2846
acc1 = predict_model(y, test, mach1)

# ╔═╡ 77449dee-602e-11eb-3d23-6baa12c216c1
md"""
### Results for least squares implementation

Good accuracy, as expected. Very high memory consumption, as expected.
The running time is quite high, but nothing to really worry about.
"""

# ╔═╡ 6e868db0-6002-11eb-0cc1-edf2da791ccb
mach2 = train_svm_hp(Xstd, y, train);

# ╔═╡ c5477f70-6001-11eb-3a08-87356dfbbd29
@benchmark predict_model($y, $test, $mach2)

# ╔═╡ c53166f2-6001-11eb-19d9-5d104a8e7107
acc2 = predict_model(y, test, mach2)

# ╔═╡ a75bd45c-602e-11eb-3206-8b097f43be6c
md"""
### Results for `libSVM` implementation

Good accuracy, as expected. Low memory consumption, as expected.

Now, regarding the running time, this is definetely what I was expecting
on the _training_ step. We can clearly see, by comparing the median time between
both benchmarks, that the `libSVM` implementation is almost **twice** as fast
as the least squares implementation.

If it is using less data, that means less computations. But also, recall that this
solver (and code) is very optimized.

## Conclusions

Most of the results presented here represent a baseline benchmark result on a simple
multiclass classification problem.

The results are expected in most cases. Although it is good to see the
`LeastSquaresSVM` doing well on the running time, both in _training_ and in
_prediction._
"""

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
# ╟─f6c9278a-6009-11eb-300b-1fb5f3a9ad13
# ╠═5e3c2068-5ffe-11eb-3173-59eb4527939f
# ╟─da5d6972-602b-11eb-3cce-a50ed90928e0
# ╠═5e286ffa-5ffe-11eb-22c3-2bba58505b97
# ╟─abe9a28a-602c-11eb-18a9-493b4b05d461
# ╟─00fdae46-601a-11eb-1db6-e73e0f4ead1e
# ╠═5e0dc6e6-5ffe-11eb-0b25-8fa52bd3cfc1
# ╟─61c5201a-602e-11eb-15a7-1f0334dfa695
# ╠═66054bac-6002-11eb-063c-5525eb2a0ca1
# ╠═c55d8cb4-6001-11eb-2e6b-655ac8539748
# ╠═949df2e8-6002-11eb-14cb-e5a269cf2846
# ╟─77449dee-602e-11eb-3d23-6baa12c216c1
# ╠═6e868db0-6002-11eb-0cc1-edf2da791ccb
# ╠═c5477f70-6001-11eb-3a08-87356dfbbd29
# ╠═c53166f2-6001-11eb-19d9-5d104a8e7107
# ╟─a75bd45c-602e-11eb-3206-8b097f43be6c
