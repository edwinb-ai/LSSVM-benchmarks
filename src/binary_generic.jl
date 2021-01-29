### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 88071c54-619d-11eb-3007-7db2c4a86215
begin
	using MLJ
	using LeastSquaresSVM
	using MLJModels
	using DataFrames
	using StatsPlots
	using BenchmarkTools
end

# ╔═╡ ae6803ac-619f-11eb-0d67-cb1be3400e9c
@load SVC pkg = LIBSVM;

# ╔═╡ b1598d44-619d-11eb-1efa-eb785775d940
gr();

# ╔═╡ cb3ab102-619d-11eb-1339-81cf03974fee
X, y = MLJ.make_moons(250; noise=0.3);

# ╔═╡ d7e2967c-619d-11eb-0d7f-9f0e79b48902
dfCircles = DataFrame(X);

# ╔═╡ e37d9ae0-619d-11eb-2ff0-3394d6447c9f
dfCircles.y = y;

# ╔═╡ 15c33f64-619e-11eb-0745-8fe5b6cf13a9
first(dfCircles, 3)

# ╔═╡ d0318284-619e-11eb-3fa1-59fb348e5d06
d = MLJ.int(y; type=Int);

# ╔═╡ 6db3e5d8-619f-11eb-1512-4d23ffd7adb3
dfCircles.c = d;

# ╔═╡ e7202e7e-619d-11eb-1b4a-e77c27e2d200
@df dfCircles scatter(:x1, :x2, colour=:c)

# ╔═╡ 3ca54cf6-619e-11eb-1a89-d7e83e9de721
function train_lssvm_hp(X, y, train)
	model = LSSVClassifier()

	r1 = range(model, :σ, lower=1e-2, upper=10.0)
	r2 = range(model, :γ, lower=1, upper=200.0)

	self_tuning_model = TunedModel(
		model=model,
		tuning=RandomSearch(),
		resampling=CV(nfolds=5),
		range=[r1, r2],
        measure=accuracy,
        n=1000
	)

	mach = machine(self_tuning_model, X, y)
	fit!(mach, rows=train)

	return mach
end;

# ╔═╡ a17664ba-619f-11eb-317c-79e5a717faca
function train_svm_hp(X, y, train)
	model = SVC()

	r1 = range(model, :cost, lower=1, upper=1000.0)
	r2 = range(model, :gamma, lower=1e-2, upper=10.0)

	self_tuning_model = TunedModel(
		model=model,
		tuning=RandomSearch(),
		resampling=CV(nfolds=5),
		range=[r1, r2],
        measure=accuracy,
        n=1000
	)

	mach = machine(self_tuning_model, X, y)
	fit!(mach, rows=train)

	return mach
end;

# ╔═╡ a6aa28f2-619f-11eb-0f6a-9167af8e877a
Xstd = MLJ.transform(MLJ.fit!(MLJ.machine(Standardizer(), X)), X);

# ╔═╡ e03600a0-619f-11eb-15a2-3f4677c2f723
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true);

# ╔═╡ 67d60f1a-61a4-11eb-3d38-35c2ccbd2744
@benchmark train_lssvm_hp($Xstd, $y, $train)

# ╔═╡ 70b7286c-61a4-11eb-28d0-4fbb79a7f373
@benchmark train_svm_hp($Xstd, $y, $train)

# ╔═╡ c1e62e68-619f-11eb-32b0-a744377d1b3c
mach1 = train_lssvm_hp(Xstd, y, train);

# ╔═╡ 2792dda8-619e-11eb-0da5-9712e7cb14de
mach2 = train_svm_hp(Xstd, y, train);

# ╔═╡ c4f54008-61a4-11eb-3b9e-994353e3ff5b
function predict_model(y, test, mach)
	results = MLJ.predict(mach, rows=test)
	acc = MLJ.accuracy(results, y[test])

	return acc
end;

# ╔═╡ 3689b474-61a5-11eb-0c60-a5c7c2a8e51e
acc1 = predict_model(y, test, mach1)

# ╔═╡ 4d1edf3c-61a5-11eb-30a9-3dc0a13fcba0
acc2 = predict_model(y, test, mach2)

# ╔═╡ 53c88e0c-61a5-11eb-3581-852491d59ffb


# ╔═╡ f9e794a6-61a3-11eb-316c-21c05947feaa


# ╔═╡ f6c26bfc-61a3-11eb-240c-97f3e8e48d63


# ╔═╡ e1dfbd84-61a3-11eb-2f44-f7146ce3045e


# ╔═╡ dbaacbf2-61a3-11eb-25b6-d33f4d04a74d


# ╔═╡ d449e050-61a3-11eb-1087-29a8e7b2b7dc


# ╔═╡ 03119e10-619e-11eb-28c6-ef029e684719


# ╔═╡ fcdf5636-619d-11eb-3019-953155bd46d3


# ╔═╡ b55b6cc8-619d-11eb-2829-7dbc5071a4e4


# ╔═╡ b544cca2-619d-11eb-30a5-c95dcb2294ea


# ╔═╡ b52c910a-619d-11eb-28af-b52571b41a82


# ╔═╡ b5135dd4-619d-11eb-226b-ad3792f806d6


# ╔═╡ b4fbe730-619d-11eb-294b-d100ca550814


# ╔═╡ b4e0b232-619d-11eb-3807-898664a71425


# ╔═╡ b4c5fc4c-619d-11eb-1567-650d59a0de08


# ╔═╡ b4abcbec-619d-11eb-3a17-b9064e2eedbc


# ╔═╡ b491fa8c-619d-11eb-3863-3ddd7fc0aa54


# ╔═╡ b4768d88-619d-11eb-0500-d3d24f8dbc5e


# ╔═╡ b45e3d8c-619d-11eb-081c-b9b953986714


# ╔═╡ b442d9b4-619d-11eb-36b5-81efc98e22cd


# ╔═╡ b42a464e-619d-11eb-11cb-870135a4f5a9


# ╔═╡ b40e3044-619d-11eb-0f41-8b657055d962


# ╔═╡ b3f4735c-619d-11eb-1610-057c920d8c69


# ╔═╡ b3a6646e-619d-11eb-1f5e-17020f493ab5


# ╔═╡ Cell order:
# ╠═88071c54-619d-11eb-3007-7db2c4a86215
# ╠═ae6803ac-619f-11eb-0d67-cb1be3400e9c
# ╠═b1598d44-619d-11eb-1efa-eb785775d940
# ╠═cb3ab102-619d-11eb-1339-81cf03974fee
# ╠═d7e2967c-619d-11eb-0d7f-9f0e79b48902
# ╠═e37d9ae0-619d-11eb-2ff0-3394d6447c9f
# ╠═15c33f64-619e-11eb-0745-8fe5b6cf13a9
# ╠═d0318284-619e-11eb-3fa1-59fb348e5d06
# ╠═6db3e5d8-619f-11eb-1512-4d23ffd7adb3
# ╠═e7202e7e-619d-11eb-1b4a-e77c27e2d200
# ╠═3ca54cf6-619e-11eb-1a89-d7e83e9de721
# ╠═a17664ba-619f-11eb-317c-79e5a717faca
# ╠═a6aa28f2-619f-11eb-0f6a-9167af8e877a
# ╠═e03600a0-619f-11eb-15a2-3f4677c2f723
# ╠═67d60f1a-61a4-11eb-3d38-35c2ccbd2744
# ╠═70b7286c-61a4-11eb-28d0-4fbb79a7f373
# ╠═c1e62e68-619f-11eb-32b0-a744377d1b3c
# ╠═2792dda8-619e-11eb-0da5-9712e7cb14de
# ╠═c4f54008-61a4-11eb-3b9e-994353e3ff5b
# ╠═3689b474-61a5-11eb-0c60-a5c7c2a8e51e
# ╠═4d1edf3c-61a5-11eb-30a9-3dc0a13fcba0
# ╠═53c88e0c-61a5-11eb-3581-852491d59ffb
# ╠═f9e794a6-61a3-11eb-316c-21c05947feaa
# ╠═f6c26bfc-61a3-11eb-240c-97f3e8e48d63
# ╠═e1dfbd84-61a3-11eb-2f44-f7146ce3045e
# ╠═dbaacbf2-61a3-11eb-25b6-d33f4d04a74d
# ╠═d449e050-61a3-11eb-1087-29a8e7b2b7dc
# ╠═03119e10-619e-11eb-28c6-ef029e684719
# ╠═fcdf5636-619d-11eb-3019-953155bd46d3
# ╠═b55b6cc8-619d-11eb-2829-7dbc5071a4e4
# ╠═b544cca2-619d-11eb-30a5-c95dcb2294ea
# ╠═b52c910a-619d-11eb-28af-b52571b41a82
# ╠═b5135dd4-619d-11eb-226b-ad3792f806d6
# ╠═b4fbe730-619d-11eb-294b-d100ca550814
# ╠═b4e0b232-619d-11eb-3807-898664a71425
# ╠═b4c5fc4c-619d-11eb-1567-650d59a0de08
# ╠═b4abcbec-619d-11eb-3a17-b9064e2eedbc
# ╠═b491fa8c-619d-11eb-3863-3ddd7fc0aa54
# ╠═b4768d88-619d-11eb-0500-d3d24f8dbc5e
# ╠═b45e3d8c-619d-11eb-081c-b9b953986714
# ╠═b442d9b4-619d-11eb-36b5-81efc98e22cd
# ╠═b42a464e-619d-11eb-11cb-870135a4f5a9
# ╠═b40e3044-619d-11eb-0f41-8b657055d962
# ╠═b3f4735c-619d-11eb-1610-057c920d8c69
# ╠═b3a6646e-619d-11eb-1f5e-17020f493ab5
