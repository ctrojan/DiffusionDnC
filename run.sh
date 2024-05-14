#!/usr/bin/bash

if [[ "$1" == "toylogreg" ]] || [[ "$1" == "all" ]]; then
	echo "Toy logistic regression"
	python3 ./experiments/toy_logistic_regression/toylogreg_setup.py
	python3 ./experiments/toy_logistic_regression/toylogreg_diffusion.py
	python3 ./experiments/toy_logistic_regression/toylogreg_gp.py
	Rscript ./experiments/toy_logistic_regression/toylogreg_semikde.R
	python3 ./experiments/toy_logistic_regression/toylogreg_comparison.py	
fi

if [[ "$1" == "mog" ]] || [[ "$1" == "all" ]]; then
	echo "Mixture of Gaussians"
	python3 ./experiments/mixture_of_gaussians/mog_setup.py
	python3 ./experiments/mixture_of_gaussians/mog_diffusion.py
	python3 ./experiments/mixture_of_gaussians/mog_gp.py
	Rscript ./experiments/mixture_of_gaussians/mog_semikde.R
	python3 ./experiments/mixture_of_gaussians/mog_comparison.py	
fi

if [[ "$1" == "powerplant" ]] || [[ "$1" == "all" ]]; then
	echo "Combined cycle power plant"
	python3 ./experiments/powerplant/powerplant_setup.py
	python3 ./experiments/powerplant/powerplant_diffusion.py
	python3 ./experiments/powerplant/powerplant_gp.py
	Rscript ./experiments/powerplant/powerplant_semikde.R
	python3 ./experiments/powerplant/powerplant_comparison.py	
fi

if [[ "$1" == "spambase" ]] || [[ "$1" == "all" ]]; then
	echo "Spambase"
	python3 ./experiments/spambase/spambase_setup.py
	python3 ./experiments/spambase/spambase_diffusion.py
	python3 ./experiments/spambase/spambase_gp.py
	Rscript ./experiments/spambase/spambase_semikde.R
	python3 ./experiments/spambase/spambase_comparison.py	
fi

if [[ "$1" == "bigtoylogreg" ]] || [[ "$1" == "all" ]]; then
	echo "Higher dimensional logistic regression"
	python3 ./experiments/toy_logistic_regression_bigger/bigtoylogreg_setup.py
	python3 ./experiments/toy_logistic_regression_bigger/bigtoylogreg_diffusion.py
	python3 ./experiments/toy_logistic_regression_bigger/bigtoylogreg_gp.py
	Rscript ./experiments/toy_logistic_regression_bigger/bigtoylogreg_semikde.R
	python3 ./experiments/toy_logistic_regression_bigger/bigtoylogreg_comparison.py	
fi
