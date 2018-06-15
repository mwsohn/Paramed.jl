# Paramed

Paramed is a port of the eponyous Stata program to conduct causal mediation analysis using parametric regression models proposed by Tyler VanderWeele. The original program is available from https://cdn1.sph.harvard.edu/wp-content/uploads/sites/603/2018/03/MediationPsychMethods.zip.

## Installation

```
  Pkg.clone("https://github.com/mwsohn/Paramed.jl")
```

## Syntax

paramed(yvar::Symbol,avar::Symbol, mvar::Symbol, a0::Int, a1::Int, m::Int, df::DataFrame; interaction::Bool = true,controlvars = [], logfile::IOStream = nothing)

### Options

#### Positional parameters

The following parameters must be provided in the order in which they are listed below.

- `yvar` - dependent variable (Symbol)
- `avar` - analytic variable (Symbol)
- `mvar` - mediator variable (Symbol)
- `a0` - natural level of the treatment (exposure)
- `a1` - alterantive treatment (exposure) level
- `m` - the level of mediator at which the controlled direct effect is to be estimated.
      If there is no treatment-mediator interaction, the controlled direct effect is the same at all levels
      and so an arbitary value can be chosen.
- `df` - DataFrame containing the analytic data

#### Keyword parameters

- `interaction` - A Boolean indicating whether treatment-mediation analysis is to be included (default: `true`)
- `controlvars` - An array of Symbols for control variables
