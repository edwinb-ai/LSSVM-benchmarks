### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 01db1fc8-5ffd-11eb-0dae-cda0ff89e63d
begin
	using MLJ, MLJModels
	using Random
	using StatsPlots
	using BenchmarkTools
	using DataFrames
	using LeastSquaresSVM
end

# ╔═╡ 143d4bd4-5ffe-11eb-23dc-1751a1913a05
@load SVC pkg=LIBSVM

# ╔═╡ 218d4792-5ffd-11eb-3cb8-11fc6c7c2d2c
gr();

# ╔═╡ 281afffa-5ffd-11eb-24ce-219f746515da
X, y = @load_iris;

# ╔═╡ 6e9dcf16-5ffd-11eb-0523-e1cfd36fd57a
dfIris = DataFrame(X);

# ╔═╡ 82244ce8-5ffd-11eb-003d-23214d962d20
dfIris.y = y;

# ╔═╡ 878df3be-5ffd-11eb-2b19-7f74692d0189
first(dfIris, 3)

# ╔═╡ 2bbc10ce-6000-11eb-2162-398983b5cf48
train, test = partition(eachindex(y), 0.6, shuffle=true);

# ╔═╡ 92b48212-5ffd-11eb-3032-d1c9519703f2
function train_lssvm(X, y, train)
	model = LSSVClassifier()
	
	r1 = range(model, :σ, lower=1e-2, upper=10.0)
	r2 = range(model, :γ, lower=1, upper=150)
	
	self_tuning_model = TunedModel(
		model=model,
		tuning=RandomSearch(),
		resampling=CV(nfolds=5),
		range=[r1, r2],
		measure=accuracy
	)
	
	mach = machine(self_tuning_model, X, y)
	fit!(mach, rows=train, verbosity=0)
	
	return mach
end;

# ╔═╡ 5973126c-5ffe-11eb-2219-85731b9c0786
function train_svm(X, y, train)
	model = SVC()
	
	r1 = range(model, :cost, lower=1, upper=1000.0)
	r2 = range(model, :gamma, lower=1e-2, upper=10.0)
	
	self_tuning_model = TunedModel(
		model=model,
		tuning=RandomSearch(),
		resampling=CV(nfolds=5),
		range=[r1, r2],
		measure=accuracy
	)
	
	mach = machine(self_tuning_model, X, y)
	fit!(mach, rows=train, verbosity=0)
	
	return mach
end;

# ╔═╡ 5e76126c-5ffe-11eb-2b0b-6fe6136909c5
Xstd = MLJ.transform(MLJ.fit!(MLJ.machine(Standardizer(), X)), X);

# ╔═╡ 5e3c2068-5ffe-11eb-3173-59eb4527939f
@benchmark train_lssvm($Xstd, $y, $train)

# ╔═╡ 66054bac-6002-11eb-063c-5525eb2a0ca1
mach1 = train_lssvm(Xstd, y, train);

# ╔═╡ 5e286ffa-5ffe-11eb-22c3-2bba58505b97
@benchmark train_svm($Xstd, $y, $train)

# ╔═╡ 6e868db0-6002-11eb-0cc1-edf2da791ccb
mach2 = train_svm(Xstd, y, train);

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
# ╠═01db1fc8-5ffd-11eb-0dae-cda0ff89e63d
# ╠═143d4bd4-5ffe-11eb-23dc-1751a1913a05
# ╠═218d4792-5ffd-11eb-3cb8-11fc6c7c2d2c
# ╠═281afffa-5ffd-11eb-24ce-219f746515da
# ╠═6e9dcf16-5ffd-11eb-0523-e1cfd36fd57a
# ╠═82244ce8-5ffd-11eb-003d-23214d962d20
# ╠═878df3be-5ffd-11eb-2b19-7f74692d0189
# ╠═2bbc10ce-6000-11eb-2162-398983b5cf48
# ╠═92b48212-5ffd-11eb-3032-d1c9519703f2
# ╠═5973126c-5ffe-11eb-2219-85731b9c0786
# ╠═5e76126c-5ffe-11eb-2b0b-6fe6136909c5
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
