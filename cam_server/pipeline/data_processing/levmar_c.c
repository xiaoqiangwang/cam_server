#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

#include "levmar-2.6/levmar.h"

static char *message[] = {
    "unkown",
    "stopped by small gradient J^T e",
    "stopped by small Dp",
    "stopped by itmax",
    "singular matrix. Restart from current p with increased mu",
    "no further error reduction is possible. Restart with increased mu",
    "stopped by small ||e||_2",
    "stopped by invalid (i.e. NaN or Inf) \"func\" values. This is a user error"
};
 
void gaussian(double *p, double *x, int m, int n, void *data)
{
    double *cdata = (double *)data;
#pragma omp parallel for simd
    for(int i=0; i<n; i++)
        x[i] = p[0] + p[1] * exp(-pow(cdata[i]-p[2],2)/(2 * pow(p[3],2)));
}

void gaussian_derive(double *p, double *jac, int m, int n, void *data)
{
    double *cdata = (double *)data;
#pragma omp parallel for simd
    for(int i=0; i<n; i++) {
        double tmp1 = pow(cdata[i] - p[2], 2) / pow(p[3], 2);
        double fac = exp(-tmp1/2);
        int j = i*4;
	double tmp2 = p[1] * fac * tmp1;

        jac[j++] = 1.0;
        jac[j++] = fac;
        jac[j++] = tmp2 / (cdata[i] - p[2]);
        jac[j++] = tmp2 / p[3];
    }
}


void _gauss_fit(double *profile, double *axis, double *p0, int n, int m, int maxfev, double opts[], double info[], int use_jac)
{
    if (use_jac == 0)
        dlevmar_dif(gaussian, 
                p0, profile, m, n,
                maxfev,
                opts,
                info,
                0, 0,
                axis);
    else
        dlevmar_der(gaussian,  gaussian_derive,
                p0, profile, m, n,
                maxfev,
                opts,
                info,
                0, 0,
                axis);
}

static PyObject *gauss_fit(PyObject *self, PyObject *args, PyObject *kws) {
    const char *kwlist[] = {"x", "y", "p0", "maxfev",  NULL};
    PyArrayObject *xval, *yval, *p0;
    int maxfev = 100;
    npy_intp *dimx, *dimp;
    int ndx, ndp, nrows, ncols;
    double *x, *y, *p;
    double opts[LM_OPTS_SZ] = {LM_INIT_MU, 1E-15, 1E-15, 1E-20, -LM_DIFF_DELTA};
    npy_intp shape[2] = {1, -1};
    PyArray_Dims dims;
    dims.ptr = shape;
    dims.len = sizeof(shape);

    if (!PyArg_ParseTupleAndKeywords(args, kws, "O!O!O!|i", (char **)kwlist, &PyArray_Type, &xval, &PyArray_Type, &yval, &PyArray_Type, &p0, &maxfev))
        return NULL;

    if (PyArray_TYPE(xval) != NPY_DOUBLE || PyArray_TYPE(yval) != NPY_DOUBLE || PyArray_TYPE(p0) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Argument inputs are not of type \"double\".");
        return NULL;
    }
    
    if (!PyArray_SAMESHAPE(xval, yval)) {
        PyErr_SetString(PyExc_AssertionError, "x, y must have the same shape");
    }

    ndx = PyArray_NDIM(xval);
    ndp = PyArray_NDIM(p0);

    dimx = PyArray_SHAPE(xval);
    dimp = PyArray_SHAPE(p0);

    if (ndx == 1) {
        nrows = 1;
        ncols = dimx[0];
        if ( ndp != 1 || dimp[0] != 4 ) {
            PyErr_SetString(PyExc_AssertionError, "Length of initial parameter vector not equal to 4.\
                  \nThe gaussian function is of form y = a + b * exp(-(x-c)**2/(2*d**2)).");
            return NULL;
        }
    }
    else if (ndx == 2) {
        nrows = dimx[0];
        ncols = dimx[1];
        if ( ndp != 2 || dimp[0] != dimx[0] || dimp[1] !=4 ) {
            PyErr_SetString(PyExc_AssertionError, "Length of initial parameter vector not equal to 4.\
                  \nThe gaussian function is of form y = a + b * exp(-(x-c)**2/(2*d**2)).");
            return NULL;
        }
    }
    else {
        PyErr_SetString(PyExc_AssertionError, "The dimension of the function input(s) must equal to 1 or 2");
        return NULL;
    }


    x = (double *) PyArray_DATA(xval);
    y = (double *) PyArray_DATA(yval);
    p = (double *) PyArray_DATA(p0);

    Py_BEGIN_ALLOW_THREADS

#pragma omp parallel for
    for (int i=0; i<nrows; i++) {
        double info[LM_INFO_SZ];
        _gauss_fit(y+i*ncols, x+i*ncols, p+i*4, ncols, 4, maxfev, opts, info, 1);
    }
    Py_END_ALLOW_THREADS

    return Py_None;
}
 
static PyMethodDef methods[] = {
    {"gauss_fit", (PyCFunction)gauss_fit, METH_VARARGS|METH_KEYWORDS, "Gaussian fit with LevMar"},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "levmar_c",
    "LevMar C funtions",
    -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_levmar_c(void)
{
    PyObject *pModule = PyModule_Create(&module);

    import_array();

    return pModule;
}
