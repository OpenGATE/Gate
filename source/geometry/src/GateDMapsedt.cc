/*******************************************************
 * david.coeurjolly@liris.cnrs.fr
 * 
 * 
 * This software is a computer program whose purpose is to [describe
 * functionalities and technical features of your software].
 * 
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 *  * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability. 
 * 
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or 
 * data to be ensured and,  more generally, to use and operate it in the 
 * same conditions as regards security. 

 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
*******************************************************/
/** region_sedt : multi-region SEDT  in linear time
 * 
 * David Coeurjolly (david.coeurjolly@liris.cnrs.fr) - Sept. 2004
 * 
 *  Feb. 2005
 *  Dec. 2007
**/

#include "GateDMapsedt.h"


////////// Functions F and Sep for the SDT labelling
/** 
 **************************************************
 * @b F
 * @param x 
 * @param i 
 * @param gi2 
 * @return Definition of a parabola
 **************************************************/
inline long F(int x, int i, long gi2)
{
  return (x-i)*(x-i) + gi2;
}

/** 
 **************************************************
 * @b Sep
 * @param i 
 * @param u 
 * @param gi2 
 * @param gu2 
 * @return The abscissa of the intersection point between two parabolas
 **************************************************/
inline long Sep(int i, int u, long gi2, long gu2) 
{
  return  (u*u - i*i + gu2 - gi2) / 2*(u-i);
}
/////////


/** 
 **************************************************
 * @b phaseSaitoX
 * @param V Input volume
 * @param sdt_x SDT along the x-direction
 **************************************************/
//First step of  the saito  algorithm 
// (Warning   : we  store the  EDT instead of the SDT)
void phaseSaitoX(const Vol &V, Longvol &sdt_x)
{
  for (int z = 0; z < V.sizeZ() ; z++) 	    
    for (int y = 0; y < V.sizeY() ; y++) 
      {

	//Voxel borders are at distance 0
	sdt_x(0,y,z)=0;
	  
	// Forward scan
	for (int x = 1; x < V.sizeX()-1 ; x++) 	    
	  if (V(x,y,z) == V(x+1,y,z))
	    sdt_x(x,y,z)= 1 + sdt_x(x-1,y,z);
	  else
	    {
	      sdt_x(x,y,z) = 0;
	      sdt_x(x+1,y,z) = 0;
	      x++;
	    }

	//Last voxel = border
	sdt_x(V.sizeX()-1,y,z) = 0;
	
	//Backward scan
	for (int x = V.sizeX() -2; x >= 0; x--)      
	  if (sdt_x(x+1,y,z) < sdt_x(x,y,z)) 
	    sdt_x(x,y,z)=1 + sdt_x(x+1,y,z);
      }
}

/** 
 **************************************************
 * @b phaseSaitoY
 * @param sdt_x the SDT along the x-direction
 * @param sdt_xy the SDT in the xy-slices
 **************************************************/
//Second      Step   of    the       saito   algorithm    using    the
//[Meijster/Roerdnik/Hesselink] optimization
void phaseSaitoY(const Vol &V,Longvol &sdt_x, Longvol &sdt_xy)
{
  
  int * s = new int[sdt_x.sizeY()]; //Center of the upper envelope parabolas
  int * t = new int[sdt_x.sizeY()]; //Separating index between 2 upper envelope parabolas 
  int q; 
  int w;

  for ( int z = 0; z<sdt_x.sizeZ(); z++) 	    
    for ( int x = 0; x < sdt_x.sizeX(); x++) 
      {
	q=0;
	s[0] = 0;
	t[0] = 0;

	//We detect border voxels before the minimization
	for (int u=1; u < sdt_x.sizeY()-1 ; u++) 
	  {
	    if (V(x,u,z) != V(x,u+1,z))
	      {
		//Border voxels detected  -> 0
		sdt_x(x,u,z)= 0;
		sdt_x(x,u+1,z)= 0;
		u++;
	      }
	  }
	//Y border voxels
	sdt_x(x,0,z) = 0;
	sdt_x(x,V.sizeY()-1,z) = 0;
	
	
	//Forward Scan
	for (int u=1; u < sdt_x.sizeY() ; u++) 
	  {
	    while ((q >= 0) &&
		   (F(t[q],s[q],sdt_x(x,s[q],z)*sdt_x(x,s[q],z)) > 
		    F(t[q],u,sdt_x(x,u,z)*sdt_x(x,u,z)))
		   ) 
	      q--;
	    
	    if (q<0) 
	      {
		q=0;
		s[0]=u;
	      }
	    else 
	      {
		w = 1 + Sep(s[q],
			    u,
			    sdt_x(x,s[q],z)*sdt_x(x,s[q],z),
			    sdt_x(x,u,z)*sdt_x(x,u,z));
	
		if (w < sdt_x.sizeY()) 
		  {
		    q++;
		    s[q]=u;
		    t[q]=w;
		  }
	      }
	  }

	//Backward Scan
	for (int u = sdt_x.sizeY()-1; u >= 0; --u) 
	  {
	    sdt_xy(x,u,z) = F(u,s[q],sdt_x(x,s[q],z)*sdt_x(x,s[q],z));	      
	    if (u==t[q]) 
	      q--;
	  }
      }
}

/** 
 **************************************************
 * @b phaseSaitoZ
 * @param sdt_xy the SDT in the xy-slices
 * @param sdt_xyz the final SDT
 **************************************************/
//Third   Step      of     the    saito   algorithm     using      the
//[Meijster/Roerdnik/Hesselink] optimization
void phaseSaitoZ(const Vol &V,Longvol &sdt_xy, Longvol &sdt_xyz)
{
  
  int * s = new int[sdt_xy.sizeZ()]; //Center of the upper envelope parabolas
  int * t = new int[sdt_xy.sizeZ()]; //Separating index between 2 upper envelope parabolas 
  int q; 
  int w;

  for ( int y = 0; y<sdt_xy.sizeY(); y++) 	    
    for ( int x = 0; x < sdt_xy.sizeX(); x++) 
      {
	q=0;
	s[0] = 0;
	t[0] = 0;
	
	//We detect border voxels before the min computation
	for (int u=1; u < sdt_xy.sizeZ()-1 ; u++) 
	  {
	    if (V(x,y,u) != V(x,y,u+1))
	      {
		//Border voxels detected
		sdt_xy(x,y,u) = 0;
		sdt_xy(x,y,u+1) = 0;
		u++;
	      }
	  }
	//Y border voxels
	sdt_xy(x,y,0) = 0;
	sdt_xy(x,y,V.sizeZ()-1) = 0;
	

	//Forward Scan
	for (int u=1; u < sdt_xy.sizeZ() ; u++) 
	  {
	    while ((q >= 0) &&
		   (F(t[q],s[q], sdt_xy(x,y,s[q])) > 
		    F(t[q],u,sdt_xy(x,y,u)))
		   ) 
	      q--;
	    
	    if (q<0) 
	      {
		q=0;
		s[0]=u;
	      }
	    else 
	      {
		w = 1 + Sep(s[q],
			    u,
			    sdt_xy(x,y,s[q]),
			    sdt_xy(x,y,u));
	
		if (w < sdt_xy.sizeZ()) 
		  {
		    q++;
		    s[q]=u;
		    t[q]=w;
		  }
	      }
	  }

	//Backward Scan
	for (int u = sdt_xy.sizeZ()-1; u >= 0; --u) 
	  {
	    sdt_xyz(x,y,u) = F(u,s[q],sdt_xy(x,y,s[q]));	      
	    if (u==t[q]) 
	      q--;
	  }
      }
}


// int main(int argc,char **argv)
// {
//   char *sourceFile;
//   char *destFile;

//   //Parsing command line
//   if (argc == 3) {
//     sourceFile = argv[1];
//     destFile= argv[2];
//   } else    {
//     cout << "Compute the Region SEDT of a VOL file\nInput: a VOL file\nOutput: a Longvol file with the SEDT\n Usage : \n\t\t sedt  <sourcefile.vol> <destfile.longvol>\n" << endl;
//     exit(1);
//   }
  
//   //Loading of the volfile
//   Vol v(sourceFile);
//   if (!v.isOK()) {
//     fprintf( stderr, "Couldn't load \"%s\" file !\n", sourceFile );
//     return 1;
//   }
  
//   //Creating the longvol structure
//   Longvol sdt_x(v.sizeX(),v.sizeY(),v.sizeZ(),0);
//   if (!sdt_x.isOK()) {
//     fprintf( stderr, "Couldn't init the longvol structure !\n" );
//     return 1;
//   }

//   //Each volume cener is (0,0,0)x(sizeX,sizeY,sizeZ)
//   v.setVolumeCenter( v.sizeX()/2, v.sizeY()/2, v.sizeZ()/2 );
//   sdt_x.setVolumeCenter( sdt_x.sizeX()/2, sdt_x.sizeY()/2, sdt_x.sizeZ()/2 );

//   cerr << "Input Vol size: "<<  sdt_x.sizeX()<<"x"<<sdt_x.sizeY()<<"x"<< sdt_x.sizeZ()<<endl;


//   //We create the other structure
//   Longvol sdt_xy(sdt_x);

//   cerr << "Scan X...";
//   phaseSaitoX(v,sdt_x);
//   cerr << " Ok"<<endl<<"Scan Y...";
//   phaseSaitoY(v,sdt_x,sdt_xy);
//   cerr << " Ok"<<endl<<"Scan Z...";
//   phaseSaitoZ(v,sdt_xy,sdt_x); //We reuse sdt_x to store the final result!!
//   cerr << "Ok"<<endl;

//   sdt_x.dumpLongvol(destFile);
  
// //   //Print the output
// //   for(int k= sdt_x.minZ() ; k < sdt_x.maxZ() ; k++)  
// //     {
// //       for(int j= sdt_x.minY() ; j < sdt_x.maxY() ; j++)
// // 	{
// // 	  for(int i= sdt_x.minX() ; i < sdt_x.maxX() ; i++)
// // 	    printf("%ld ", sdt_x(i,j,k));
// // 	  printf("\n");
// // 	}
// //       printf("\n");
// //     }


//   exit(0);
// }
/**
 =================================================
 * @file   sedt.cc
 * @author David COEURJOLLY <David Coeurjolly <dcoeurjo@liris.cnrs.fr>>
 * @date   Wed Sep 29 17:05:31 2004
 * 
 * @brief  The Euclidean distance transform in linear time using the Saito and Toriwaki
 * algorithm with the Meijster/Roerdnik/Hesselink's optimization
 *
 * Computational cost : O(n^3) if the input volume is nxnxn
 *
 * Memory requirement : O(n^3). 
 *
 * More precisely : if nxnxn*(size of a char) is the size of the input volume, 
 * the SDT requires  nxnxn*(size of a long int). Furthermore a temporary  nxnxn*(size of a long int)
 * is needed all along the process. Two vectors with size  n*(size of an int) (arrays s and q) are also used.
 *
 =================================================*/
