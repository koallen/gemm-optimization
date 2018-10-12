#ifndef AUX_H_
#define AUX_H_

#include <stdio.h>
#define SAFE_MALLOC(call)                                               \
	{                                                               \
		int ret = (call);                                       \
		if (ret != 0)                                           \
			printf("Error at %s:%d\n", __FILE__, __LINE__); \
	}


#endif
