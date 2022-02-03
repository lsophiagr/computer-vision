import timeit
import numpy

cy = timeit.timeit('sharp', setup='import sharp', number=1)
print(cy)
py = timeit.timeit('tarea_3', setup='import tarea_3', number=1)


print(cy, py)
print('Cythons speed up es de {}'.format(py/(cy-py)))