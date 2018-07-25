validating_closure_data_aposteriori.so: validating_closure_data_aposteriori.pyx
	python setup.py build_ext --inplace

training.so: training.pyx
	python setup.py build_ext --inplace

spectral_vbe.so: spectral_vbe.pyx
	python setup.py build_ext --inplace

run_aposteriori.so: run_aposteriori.pyx
	python setup.py build_ext --inplace

prepare_data.so: prepare_data.pyx
	python setup.py build_ext --inplace

plot_result.so: plot_result.pyx
	python setup.py build_ext --inplace

gen_poly_feature.so: gen_poly_feature.pyx
	python setup.py build_ext --inplace

config.so: config.pyx
	python setup.py build_ext --inplace

main.so: main.pyx
	python setup.py build_ext --inplace

.PHONY: build clean test

test:
	python cython_main.py 2d_vdp sindy 0 tdf 3

clean:
	rm -f ./*.c
	rm -f ./*.so
	rm -f ./*.pyc
	rm -rf ./build
build: main.so 
	python setup.py build_ext --inplace


