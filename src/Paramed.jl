module Paramed

# package code goes here

using DataFrames, NamedArrays, GLM

export paramed, rmse

function vcov_stata(vcov::Array{Float64,2})
	vcov2 = vcov
	n = size(vcov,1)
	for i = 2:n
		for j = 2:n
			vcov2[i-1,j-1] = vcov[i,j]
		end
	end
	vcov2[n,n] = vcov[1,1]
	for i = 1:n-1
		vcov2[i,n] = vcov[i,1]
		vcov2[n,i] = vcov[1,i]
	end
	return vcov2
end

function rmse(obj)
	return sqrt(sum((model_response(obj) .- predict(obj)).^2)/nobs(obj))
end

function paramed(yvar::Symbol,avar::Symbol, mvar::Symbol, a0::Int, a1::Int, m::Int, df::DataFrame; interaction::Bool = true,controlvars = [], logfile::IOStream = nothing)

	# convert them to Float64
	yvarf = Symbol(string(yvar,"f"))
	df[yvarf] = Vector{Float64,Missing}(df[yvar])

	mvarf = Symbol(string(mvar,"f"))
	df[mvarf] = Vector{Float64,Missing}(df[mvar])

	# process control variables
	if length(controlvars) > 0
		# convars = controlvars[1]
		# for i = 2:length(controlvars)
		# 	convars = :($convars + $(controlvars[i]))
		# end
		# fm2 = $mvar ~ $avar + $convars
		fm2 = Stella.formula($mvarf,vcat($avar,controlvars))
	else
		# fm2 = $mvar ~ $avar
		fm2 = @eval(@formula($mvarf  ~ $avar))
	end

	# process interaction terms
	if interaction == true
		ivar = Symbol(string("_",avar,"X",mvar))
		df[ivar] = df[avar] .* df[mvarf]

		fm1 = Stella.formula($yvarf,vcat($avar,$mvarf,$ivar,controlvars))

		# if length(controlvars) > 0
		# 	# fm1 = $yvar ~ $avar + $mvar + $ivar + $convars
		# 	fm1 = Stella.formula($yvar,vcat($avar,$mvar,$ivar,controlvars))
		# else
		# 	fm1 = Stella.formula($yvar,vcat($avar,$mvar,$ivar))
		# end
	else
		fm1 = Stella.formula($yvarf,vcat($avar,$mvarf,controlvars))
		# if length(controlvars) > 0
		# 	fm1 = $yvar ~ $avar + $mvar + $convars
		# else
		# 	fm1 = $yvar ~ $avar + $mvar
		# end
	end

	if isa(df[yvar],CategoricalArray) && isa(df[mvar],CategoricalArray)
		return paramed_logit_logit(df,fm1,fm2,a0,a1,m,interaction, logfile)
	elseif isa(df[yvar],CategoricalArray) && isa(df[mvar],CategoricalArray) == false
		return paramed_logit_linear(df,fm1,fm2,a0,a1,m,interaction, logfile)
	end
end

function paramed_logit_logit(df::DataFrame,fm1::Formula,fm2::Formula,a0,a1,m,interaction,logfile)

	# print(fm1)
	logt1 = glm(fm1, df, Binomial())
	print(logfile,"\n\n",coeftable(logt1),"\n\n")

	logt2 = glm(fm2, df, Binomial())
	print(logfile,"\n\n",coeftable(logt2),"\n\n")

	# covariate mean values
	cmean = transpose(mean(logt2.mm.m,1)[3:end])

    # save variance-covariance matrices and coefficients
    vcov1 = vcov_stata(vcov(logt1))
	theta = coef(logt1)
	theta1 = theta[2] # estimate for avar
	theta2 = theta[3] # estimate for mvar
	theta3 = interaction ? theta[4] : 0. # estimate for the interaction between avar and mvar

	vcov2 = vcov_stata(vcov(logt2))
	beta = coef(logt2)
	beta0 = beta[1] # estimate for the intercept
	beta1 = beta[2] # estimate for avar
	beta2 = transpose(beta[3:end]) # estimates for all covariates in the mvar logistic model

	# model 2: estimates * mean
	cmeansum = sum(beta2*cmean')

	# zero1, zero2 - empty matrices initialized to zeros
	zero1 = zeros(size(vcov1,1),size(vcov2,1))
	zero2 = zeros(size(vcov2,1),size(vcov1,1))

	# sigma
	sigma = vcat(hcat(vcov2, zero2),hcat(zero1, vcov1))

	# Marginal CDE
	z1 = zeros(1,size(beta,1)-2)
	z = hcat(0,z1,0,1,0)
	marggammacde = hcat(z,m,z1,0)

	# Marginal NDE
	A=exp(theta2 + theta3*a1 + beta0 + beta1*a0 + cmeansum)
	B = (1+exp(theta2 + theta3*a1 + beta0 + beta1*a0 + cmeansum))
	D = exp(theta2 + theta3*a0 + beta0 + beta1*a0 + cmeansum)
	E = (1 + exp(theta2 + theta3*a0 + beta0 + beta1*a0 + cmeansum))

	d1nde = A/B - D/E
	d2nde = a0*d1nde
	d3nde = d1nde .* cmean
	d4nde = 0
	d5nde = (a1 - a0)
	d6nde = d1nde
	d7nde = a1*A/B - a0*D/E
	d8nde = z1
	marggammapnde = hcat(d2nde, d3nde, d1nde, d5nde,d6nde,d7nde,d8nde, d4nde)

	# Marginal TNDE
	A = exp(theta2 + beta0 + beta1*a1 + cmeansum)
	B = (1 + exp(theta2 + beta0 + beta1*a1 + cmeansum))
	D = exp(theta2 + beta0 + beta1*a1 + cmeansum)
	E = (1 + exp(theta2 + beta0 + beta1*a1 + cmeansum))
	s = A/B - D/E
	x = a1 * s
	w = s .* cmean
	t = a1 - a0
	h=a1*A/B-a0*D/E
	marggammatnde = hcat(x, w, s,  t, s, h, z1, 0)

	# Marginal TNIE
	A = exp(theta2 + beta0 + beta1*a1 + cmeansum)
	B = (1 + exp(theta2 + beta0 + beta1*a1 + cmeansum))
	D = exp(theta2 + beta0 + beta1*a0 + cmeansum)
	E = (1 + exp(theta2 + beta0 + beta1*a0 + cmeansum))
	F = exp(beta0 + beta1*a0 + cmeansum)
	G = (1 + exp(beta0 + beta1*a0 + cmeansum))
	H = exp(beta0 + beta1*a1 + cmeansum)
	I = (1 + exp(beta0 + beta1*a1 + cmeansum))
	d1nie = F/G - H/I + A/B - D/E
	d2nie = a0*F/G - a1*H/I + a1*A/B - a1*D/E
	d3nie = cmean.*d1nie
	d4nie = 0
	d5nie = 0
	d6nie = A/B - D/E
	d7nie = a1*(A/B - D/E)
	d8nie = z1
	marggammatnie = hcat(d2nie, d3nie, d1nie,  d5nie, d6nie, d7nie, d8nie, d4nie)

	# Marginal PNIE
	A = exp(theta2 + beta0 + beta1*a1 + cmeansum)
	B = (1+exp(theta2 + beta0 + beta1*a1 + cmeansum))
	D = exp(theta2 + beta0 + beta1*a0 + cmeansum)
	E = (1 + exp(theta2 + beta0 + beta1*a0 + cmeansum))
	F = exp(beta0 + beta1*a0 + cmeansum)
	G = (1 + exp(beta0 + beta1*a0 + cmeansum))
	H = exp(beta0 + beta1*a1 + cmeansum)
	I = (1 + exp(beta0 + beta1*a1 + cmeansum))
	s = F/G - H/I + A/B - D/E
	x = a0*F/G - a1*H/I + a1*A/B - a1*D/E
	w = cmean.*s
	l = A/B - D/E
	k = a0*(A/B - D/E)
	marggammapnie = hcat(x, w, s,  0, l, k, z1, 0)

	# marg se cde
	intse6=sqrt(marggammacde*sigma*marggammacde')[1,1]

	# marg se pnde
	intse7=sqrt(marggammapnde*sigma*marggammapnde')[1,1]

	# marg se pnie
	intse8=sqrt(marggammapnie*sigma*marggammapnie')[1,1]

	# marg se tnde
	intse9=sqrt(marggammatnde*sigma*marggammatnde')[1,1]

	# marg se tnie
	intse10=sqrt(marggammatnie*sigma*marggammatnie')[1,1]

	d1 = d1nie + d1nde
	d2 = d2nie + d2nde
	d3 = d3nie + d3nde
	d4 = d4nie + d4nde
	d5 = d5nie + d5nde
	d6 = d6nie + d6nde
	d7 = d7nie + d7nde
	tegammamarg=hcat(d2,d3,d1, d5,d6,d7,z1,d4)
	tesemarg=sqrt(tegammamarg*sigma*tegammamarg')[1,1]

	# Marginal CDE (Controlled Direct Effect)
	int6=exp((theta1+theta3*m)*(a1-a0))

	# Marginal NDE (Natural Direct Effect)
	int7=exp(theta1*(a1-a0))*(1+exp(theta2+theta3*a1+beta0+beta1*a0+cmeansum))/(1+exp(theta2+theta3*a0+beta0+beta1*a0+cmeansum))

	# Marginal NIE
	int8=((1+exp(beta0+beta1*a0+cmeansum))*(1+exp(theta2+theta3*a0+beta0+beta1*a1+cmeansum))) / ((1+exp(beta0+beta1*a1+cmeansum))*(1+exp(theta2+theta3*a0+beta0+beta1*a0+cmeansum)))

	# Marginal TNDE
	int9=exp(theta1*(a1-a0))*(1+exp(theta2+theta3*a1+beta0+beta1*a1+cmeansum))/(1+exp(theta2+theta3*a0+beta0+beta1*a1+cmeansum))

	# Marginal TNIE (Natural Indirect Effect)
	# println(theta1," ",theta2," ",theta3," ",beta0," ",beta1," ",cmeansum)
	int10=((1+exp(beta0+beta1*a0+cmeansum))*(1+exp(theta2+theta3*a1+beta0+beta1*a1+cmeansum))) / ((1+exp(beta0+beta1*a1+cmeansum))*(1+exp(theta2+theta3*a1+beta0+beta1*a0+cmeansum)))

	# Marginal Total Effects
	temarg = int7*int10
	logtemarg = log(int7*int10)

	# Proportion Mediated
	pm = int7*(int10 - 1) / (int7*int10-1)

	estse = [intse6,intse7,intse8,intse9,intse10]
	est = [int6,int7,int8,int9,int10]
	lnest = map(x->log(x),est)
	cilb  = map(x -> lnest[x] - 1.96*estse[x],1:5)
	ecilb = map(x -> exp(x), cilb)
	ciub  = map(x -> lnest[x] + 1.96*estse[x],1:5)
	eciub = map(x -> exp(x), ciub)

	# P-value and confidence intervals
	ptwosidetemarg = 2*ccdf(Normal(),logtemarg/tesemarg)
	citelmarg=exp(logtemarg-1.96*tesemarg)
	citeumarg=exp(logtemarg+1.96*tesemarg)

	ptwoside = map(x -> 2*ccdf(Normal(),lnest[x]/estse[x]),1:5)

	#
	value1 = hcat(est[1], est[2],est[5]) # int6 , int7, int10)
	se1 = hcat(estse[1], estse[2],estse[5]) #intse6, intse7, intse10)
	pvalue1 = hcat(ptwoside[1],ptwoside[2],ptwoside[5]) # 6 , ptwoside7, ptwoside10)
	cil1 = hcat(ecilb[1], ecilb[2], ecilb[5]) #ci6l,ci7l,ci10l)
	ciu1 = hcat(eciub[1], eciub[2], eciub[5]) #ci6u,ci7u,ci10u)
	x1 = hcat(value1', se1', pvalue1', cil1', ciu1')
	value2 = hcat(temarg , pm)
	se2 = hcat(tesemarg ,0)
	pvalue2 = hcat(ptwosidetemarg , 0)
	cil2 = hcat(citelmarg,0)
	ciu2 = hcat(citeumarg,0)
	x2 = hcat(value2',se2',pvalue2',cil2',ciu2')
	x = vcat(x1, x2)

	rownames = ["cde" => 1, "nde" => 2, "nie" => 3, "mte" => 4, "proportion mediated" => 5]
	colnames = ["Estimate" => 1, "SE" => 2, "P-Value" => 3, "95% CI LB" => 4, "95% CI UB" => 5]
    na = NamedArray(x, (rownames,colnames), ("Effects","Statistics") )

	# output logs
	print(logfile,"\n\n",na,"\n\n")

	return na
end

function paramed_logit_linear(df::DataFrame,fm1::Formula,fm2::Formula,a0,a1,m,interaction,logfile)

	# print(fm1)
	logt1 = glm(fm1, df, Binomial())
	print(logfile,"\n\n",coeftable(logt1),"\n\n")

	linear2 = glm(fm2, df, Normal())
	print(logfile,"\n\n",coeftable(linear2),"\n\n")

	# covariate mean values
	cmean = transpose(mean(linear2.mm.m,1)[3:end])

    # save variance-covariance matrices and coefficients
    vcov1 = vcov_stata(vcov(logt1))
	theta = coef(logt1)
	theta1 = theta[2] # estimate for avar
	theta2 = theta[3] # estimate for mvar
	theta3 = interaction ? theta[4] : 0. # estimate for the interaction between avar and mvar

	vcov2 = vcov_stata(vcov(linear2))
	beta = coef(linear2)
	s2 = rmse(linear2)
	beta0 = beta[1] # estimate for the intercept
	beta1 = beta[2] # estimate for avar
	beta2 = transpose(beta[3:end]) # estimates for all covariates in the mvar linear model

	# model 2: estimates * mean
	cmeansum = sum(theta3*beta2*cmean') # sum(beta2*cmean')

	# zero1, zero2 - empty matrices initialized to zeros
	zero1 = zeros(size(vcov1,1),size(vcov2,1))
	zero2 = zeros(size(vcov2,1),size(vcov1,1))
	z2 = zeros(size(vcov1,1),1)
	z3 = zeros(size(vcov2,1),1)

	# sigma
	sigma = vcat(hcat(vcov2, zero2, z3),hcat(zero1, vcov1, z2),hcat(zeros(1,size(vcov1,1)+size(vcov2,1)),s2))

	z1 = zeros(1,size(beta,1)-2)
	z = hcat(0,z1,0,1,0)

	# Marginal CDE
	marggammacde = hcat(z,m,z1,0,0)

	# Marginal NDE
	x = theta3*a0
	w = theta3 .* cmean'
	h = beta0 + beta1 * a0 + beta2*cmean' + theta2 * s2 + theta3 * s2 * (a1+a0)
	ts = s2*theta3
	f = theta3*theta2+0.5*(theta3^2)*(a1+a0)
	marggammapnde = hcat(x, w', theta3,  1, ts, h, z1, 0,  f)

	# Marginal TNDE
	x = theta3*a1
	w = theta3*cmean'
	h = beta0+beta1*a1+beta2*cmean'+theta2*s2+theta3*s2*(a1+a0)
	ts = s2*theta3
	f = theta3*theta2+0.5*theta3^2*(a1+a0)
	marggammatnde= hcat(x, w', theta3, 1, ts, h, z1, 0, f)

	# Marginal TNIE
	x=theta2+theta3*a1
	w=beta1*a1
	marggammatnie = hcat(x, z1, 0, 0, beta1, w, z1 , 0, 0)

	# Marginal PNIE
	x=theta2+theta3*a0
	w=beta1*a0
	marggammapnie = hcat(x, z1, 0, 0, beta1, w , z1, 0, 0)

	# marg se cde
	intse6=sqrt(marggammacde*sigma*marggammacde')[1,1]

	# marg se pnde
	intse7=sqrt(marggammapnde*sigma*marggammapnde')[1,1]

	# marg se pnie
	intse8=sqrt(marggammapnie*sigma*marggammapnie')[1,1]

	# marg se tnde
	intse9=sqrt(marggammatnde*sigma*marggammatnde')[1,1]

	# marg se tnie
	intse10=sqrt(marggammatnie*sigma*marggammatnie')[1,1]

	d2pnde=theta3*a0
	d3pnde=theta3 .* cmean
	d7pnde=beta0+beta1*a0+beta2*cmean'+theta2*s2+theta3*s2*(a1+a0)
	d6pnde=s2*theta3
	d9pnde=theta3*theta2+0.5*(theta3^2)*(a1+a0)
	d2tnie=theta2+theta3*a1
	d7tnie=beta1*a1
	d2=d2pnde+d2tnie
	d3=d3pnde
	d6=d6pnde+beta1
	d7=d7pnde+d7tnie
	d9=d9pnde
	tegammamarg=hcat(d2, d3, theta3, 1, d6, d7, z1, 0, d9)
	tesemarg=sqrt(tegammamarg*sigma*tegammamarg')[1,1]
	tsq=(theta3^2)
	rm=s2
	asq=(a1^2)
	a1sq=(a0^2)

	# Marginal CDE (Controlled Direct Effect)
	int6=exp((theta1 + theta3*m)*(a1 - a0))

	# Marginal NDE (Natural Direct Effect)
	int7 = exp((theta1 + theta3*beta0 + theta3*beta1*a0 + cmeansum + theta3*theta2*rm)*(a1-a0)+0.5*tsq*rm*(asq-a1sq))

	# Marginal NIE
	int8=exp((theta2*beta1 + theta3*beta1*a0)*(a1-a0))

	# Marginal TNDE
	int9=exp((theta1 + theta3*beta0 + theta3*beta1*a1 + cmeansum + theta3*theta2*rm)*(a1 - a0) + 0.5*tsq*rm*(asq - a1sq))

	# Marginal TNIE (Natural Indirect Effect)
	# println(theta1," ",theta2," ",theta3," ",beta0," ",beta1," ",cmeansum)
	int10=exp((theta2*beta1 + theta3*beta1*a1)*(a1 - a0))

	# Marginal Total Effects
	logtemarg = (theta1 + theta3*beta0 + theta3*beta1*a0 + cmeansum + theta2*beta1 + theta3*beta1*a1+theta3*(rm)*theta2)*(a1 - a0) + 0.5*(theta3^2)*(rm)*(a1^2 - a0^2)
	temarg = exp(logtemarg)

	# Proportion Mediated
	pm = int7*(int10 - 1) / (int7*int10-1)

	estse = [intse6,intse7,intse8,intse9,intse10]
	est = [int6,int7,int8,int9,int10]
	lnest = map(x->log(x),est)
	cilb  = map(x -> lnest[x] - 1.96*estse[x],1:5)
	ecilb = map(x -> exp(x), cilb)
	ciub  = map(x -> lnest[x] + 1.96*estse[x],1:5)
	eciub = map(x -> exp(x), ciub)

	# P-value and confidence intervals
	ptwosidetemarg = 2*ccdf(Normal(),logtemarg/tesemarg)
	citelmarg=exp(logtemarg-1.96*tesemarg)
	citeumarg=exp(logtemarg+1.96*tesemarg)

	ptwoside = map(x -> 2*ccdf(Normal(),lnest[x]/estse[x]),1:5)

	#
	value1 = hcat(est[1], est[2],est[5]) # int6 , int7, int10)
	se1 = hcat(estse[1], estse[2],estse[5]) #intse6, intse7, intse10)
	pvalue1 = hcat(ptwoside[1],ptwoside[2],ptwoside[5]) # 6 , ptwoside7, ptwoside10)
	cil1 = hcat(ecilb[1], ecilb[2], ecilb[5]) #ci6l,ci7l,ci10l)
	ciu1 = hcat(eciub[1], eciub[2], eciub[5]) #ci6u,ci7u,ci10u)
	x1 = hcat(value1', se1', pvalue1', cil1', ciu1')
	value2 = hcat(temarg , pm)
	se2 = hcat(tesemarg ,0)
	pvalue2 = hcat(ptwosidetemarg , 0)
	cil2 = hcat(citelmarg,0)
	ciu2 = hcat(citeumarg,0)
	x2 = hcat(value2',se2',pvalue2',cil2',ciu2')
	x = vcat(x1, x2)

	rownames = ["cde" => 1, "nde" => 2, "nie" => 3, "mte" => 4, "proportion mediated" => 5]
	colnames = ["Estimate" => 1, "SE" => 2, "P-Value" => 3, "95% CI LB" => 4, "95% CI UB" => 5]
    na = NamedArray(x, (rownames,colnames), ("Effects","Statistics") )
	print(logfile,"\n\n",na,"\n\n")

	return na
end

end # module
