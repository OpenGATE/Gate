/*  @(#)isotope_info.h	2.4  5/4/92  */

/*
#static char sccsid[]="@(#)isotope_info.h	2.4 5/4/92  Copyright 1989-1991 CTI PET Systems, Inc.";
*/
 	/* Isotope information */

#define NumberOfIsotopes 16

struct fixed_item{
	char *name;
	char *halflife;
	char *branch_ratio;
	float hl;
};

	/* ALWAYS add new isotopes to the END of the following array to
	   ensure backward compatibility */

static struct fixed_item isotope_info[] = {
	{ "Br-75", "98.0 min", "0.755", 5880.0},
	{ "C-11",  "20.4 min", "0.9976", 1224.0},
	{ "Cu-62",  "9.73 min", "0.980", 583.8},
	{ "Cu-64",  "12.8 hours", "0.184", 46080},
	{ "F-18",  "1.83 hours", "0.967", 6588.0},
	{ "Fe-52", "83.0 hours", "0.57", 298800.0},
	{ "Ga-68", "68.3 min", "0.891", 4098.0},
	{ "Ge-68", "275.0 days", "0.891", 23760000.0},
	{ "N-13",  "9.97 min", "0.9981", 598.2},
	{ "O-14",  "70.91 sec", "1.0", 70.91},
	{ "O-15", "123.0 sec", "0.9990", 123.0},
	{ "Rb-82", "78.0 sec", "0.950", 78.0},
	{ "Na-22", "950 days", "0.9055", 82080000.0},
	{ "Zn-62", "9.3 hours", "0.152", 33480.0},
	{ "Br-76", "16.2 hours", "0.57", 58320.0},
	{ "K-38", "7.636 min", "1.0", 458.16}
};

	/*	define some descriptive indices */

#define	BR75	0
#define	C11	1
#define	CU62	2
#define	CU64	3
#define	F18	4
#define	FE52	5
#define	GA68	6
#define	GE68	7
#define	N13	8
#define	O14	9
#define	O15	10
#define	RB82	11
#define	NA22	12
#define	ZN62	13
#define	BR76	14
#define	K38	15


