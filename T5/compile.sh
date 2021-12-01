echo "Compiling static"
gcc -o matrix_lib_test matrix_lib_test.c matrix_lib_vh.c timer.c -I/opt/nec/ve/veos/include -L/opt/nec/ve/veos/lib64 -Wl,-rpath=/opt/nec/ve/veos/lib64 -lveo
echo "Done static"