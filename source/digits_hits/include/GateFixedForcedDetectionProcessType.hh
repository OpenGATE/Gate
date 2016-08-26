#ifndef GATEFIXEDFORCEDDETECTIONPROCESSTYPE_HH
#define GATEFIXEDFORCEDDETECTIONPROCESSTYPE_HH

// Define all the processes handled by the fixed forced detection actor.
// Pruimary is not a process so it is defined at the end to serve as an
// upper limit when looping on the processes.
typedef enum {
  COMPTON=0,
  RAYLEIGH,
  PHOTOELECTRIC,
  ISOTROPICPRIMARY,
  PRIMARY,
} ProcessType;


#endif
