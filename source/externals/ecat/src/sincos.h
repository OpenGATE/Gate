
#ifndef sincos_h
#define sincos_h
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if !defined(__sun) || defined(__SVR4)       /* solaris 2.x*/

static void sincos(theta, sintheta, costheta)
double theta, *sintheta, *costheta;
{
	*sintheta = sin(theta);
	*costheta = cos(theta);
}
#endif /* sun */

#endif /* sincos_h */


