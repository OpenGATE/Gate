// ************************************************************************
// * Materials definition
// ************************************************************************
// * JB - 2011-08-05 15:15:44

// List of index materials
//  0- Air
//  1- Water
//  2- Body
//  3- Lung
//  4- Breast
//  5- Heart
//  6- SpineBone
//  7- RibBone
//  8- Intestine
//  9- Spleen
// 10- Blood
// 11- Liver
// 12- Kidney
// 13- Brain
// 14- Pancreas

// vesna - fresnel
__constant__ const float COSZERO = 0.999999f;
__constant__ const float COS90D = 1.0e-6f;
// vesna - fresnel

// Number of elements per material
__constant__ unsigned short int mat_nb_elements [15] = {
	// Air Water Body Lung Breast Heart SpineBone RibBone Intestine Spleen
	   4,  2,    2,   9,   8,     9,    11,       9,      9,        9,
	// Blood Liver Kidney Brain Pancreas
	   10,   9,    10,    9,    9
};

// vesna
// Anisotropy (g) per material
__constant__ float mat_anisotropy [15] = {
	// Air   Water   Body   Lung   Breast Heart SpineBone RibBone Intestine Spleen
	   0.5,   0.6,   0.8,    0.6,   0.7,   0.76,    0.86,     0.6,    0.64,      0.69,
	// Blood Liver Kidney Brain Pancreas
	   0.8,   0.86,  0.84,   0.78,   0.76
};

// Rindex per material
// !!! make sure we use same rindex for 2 materials when only looking at Mie scattering!!!
__constant__ float mat_Rindex [15] = {
	// Air   Water   Body   Lung   Breast Heart SpineBone RibBone Intestine Spleen
	   1.0,   1.2,   1.3,    1.4,   1.2,   1.3,    1.4,     1.6,    1.1,      1.6,
	// Blood Liver Kidney Brain Pancreas
	   1.2,   1.3,  1.4,   1.2,   1.3
};

// Mie scattering lengths:
// {Energy, scattering_length, Energy, scattering_length, Energy, scattering_length}
__constant__ float Mie_scatteringlength_Table[15][6] =
{ 
{ 5.0000E-06,  5.3000E+00, 6.0000E-06,  6.2000E+00, 7.0000E-06,  6.7000E+00  } ,
{ 5.0000E-06,  1.3000E+00, 6.0000E-06,  1.2000E+00, 7.0000E-06,  1.7000E+00  } ,
{ 5.0000E-06,  6.3000E+00, 6.0000E-06,  7.2000E+00, 7.0000E-06,  7.7000E+00  } ,
{ 5.0000E-06,  1.3000E+00, 6.0000E-06,  1.2000E+00, 7.0000E-06,  1.7000E+00  } ,
{ 5.0000E-06,  8.3000E+00, 6.0000E-06,  4.2000E+00, 7.0000E-06,  3.7000E+00  } ,
{ 5.0000E-06,  5.7000E+00, 6.0000E-06,  6.9000E+00, 7.0000E-06,  6.7000E+00  } ,
{ 5.0000E-06,  2.3000E+00, 6.0000E-06,  8.2000E+00, 7.0000E-06,  6.2000E+00  } ,
{ 5.0000E-06,  1.3000E+00, 6.0000E-06,  1.5000E+00, 7.0000E-06,  4.9000E+00  } ,
{ 5.0000E-06,  5.8000E+00, 6.0000E-06,  4.2000E+00, 7.0000E-06,  6.7000E+00  } ,
{ 5.0000E-06,  2.3000E+00, 6.0000E-06,  6.9000E+00, 7.0000E-06,  6.0000E+00  } ,
{ 5.0000E-06,  5.3000E+00, 6.0000E-06,  1.2000E+00, 7.0000E-06,  4.7000E+00  } ,
{ 5.0000E-06,  7.3000E+00, 6.0000E-06,  3.7000E+00, 7.0000E-06,  2.9000E+00  } ,
{ 5.0000E-06,  1.3000E+00, 6.0000E-06,  3.7000E+00, 7.0000E-06,  1.7000E+00  } ,
{ 5.0000E-06,  9.3000E+00, 6.0000E-06,  1.2000E+00, 7.0000E-06,  6.5000E+00  } ,
{ 5.0000E-06,  3.3000E+00, 6.0000E-06,  2.2000E+00, 7.0000E-06,  8.7000E+00  }
};
// vesna

