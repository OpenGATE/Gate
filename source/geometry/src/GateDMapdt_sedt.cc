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
/** sedt : SEDT  in linear time
 *
 * David Coeurjolly (david.coeurjolly@liris.cnrs.fr) - Sept. 2004
 *

 *
 **/

//#define _MULTITHREAD 1

#include "GateDMapdt.h"
#include "GateDMapdt_core.h"

//Inline def of +infty opertators
#include "GateDMapoperators.ihh"


/***************************************************************************************************************/
/***************************************************************************************************************/
/*                          Phase X                                                                            */
/***************************************************************************************************************/
/***************************************************************************************************************/


inline void phaseSaitoX_1D(const Vol &V, Longvol &sdt_x, const bool isMultiregion, const bool isToric, int y,int z)
{

  unsigned int index=0;     //Break Index (toric case)
  unsigned int cpt, cpt2;

  //TORIC Case We look for a min index
  index = 0;

  if (not(isMultiregion))
    {
      while ((int)index < V.sizeX() and (V(index,y,z) != 0))
        {
	  sdt_x(index,y,z)=INFTY;
	  index++;
        }
    }
  //We have found at least one region border
  if ((int)index < V.sizeX())
    {
      sdt_x(index,y,z) = 0; //Either Multiregion or isToric index
      cpt = (index + 1) % V.sizeX();
      cpt2 = index;

      // Forward scan
      unsigned int bound;
      if (isToric)
	bound = V.sizeX();  //V.sizex() cyclic steps
      else
	bound = V.sizeX() - index;

      for (unsigned int x = 1; x < bound; x++)
        {
	  if (not (isMultiregion))
            {
	      if (V(cpt,y,z) == 0)
		sdt_x(cpt,y,z) = 0;
	      else
		sdt_x(cpt,y,z) = sum(1, sdt_x(cpt2,y,z));
            }
	  else   //Multi-region case
            {
	      if (V(cpt2,y,z) == V(cpt,y,z))
		sdt_x(cpt,y,z)= sum(1, sdt_x(cpt2,y,z));
	      else
                {
		  sdt_x(cpt,y,z) = 0;
		  sdt_x(cpt2,y,z) = 0;
                }
            }
	  cpt2 = cpt;
	  cpt = (cpt +1) % V.sizeX();
        }

      //Last border
      if (isMultiregion)
	sdt_x(V.sizeX()-1,y,z) = 0;

      //Backward scan
      if (not(isToric))
	cpt = V.sizeX()-2;
      else
	cpt = (index + V.sizeX()-1) % V.sizeX();
      for (int x = V.sizeX() -2; x >= 0; x--)
        {
	  if (sdt_x((cpt+1)%V.sizeX(),y,z) < sdt_x(cpt,y,z))
	    sdt_x(cpt,y,z)=sum(1, sdt_x((cpt+1)%V.sizeX(),y,z));
	  cpt = (cpt + V.sizeX() -1 ) % V.sizeX();
        }
    }
}


#ifdef _MULTITHREAD
struct thread_data
{
  int  thread_id;
  const Vol *V;
  Longvol *sdt_x;
  Longvol *output;
  bool isMultiregion;
  bool isToric;
  int minIter;
  int maxIter;
};

void *phaseSaitoX_blockZ_multithread(void *threadarg)
{
  struct thread_data *data;
  data = (struct thread_data* ) threadarg;


  cout << "Thread ("<<data->thread_id<<"): "<< data->minIter<<" -> "<<data->maxIter<<endl;

  for (int z = data->minIter; z < data->maxIter ; z++)
    for (int y = 0; y < data->V->sizeY() ; y++)
      {
	phaseSaitoX_1D(*(data->V),*(data->sdt_x),data->isMultiregion,data->isToric,y,z);
      }
  return 0;
}
#endif

//--------------------------------------------------------------------
void phaseSaitoX_blockZ(const Vol &V, Longvol &sdt_x, const bool isMultiregion, const bool isToric, int minZ,int maxZ)
{
  for (int z = minZ; z < maxZ ; z++)
    for (int y = 0; y < V.sizeY() ; y++)
      {
	phaseSaitoX_1D(V,sdt_x,isMultiregion,isToric,y,z);
      }
}


/**
**************************************************
* @b phaseSaitoX
* @param V Input volume
* @param sdt_x SDT along the x-direction
**************************************************/
//First step of  the saito  algorithm
// (Warning   : we  store the  EDT instead of the SDT)
void phaseSaitoX(const Vol &V, Longvol &sdt_x, 
                 const bool isMultiregion, 
                 const bool isToric, 
                 const int /*NbThreads=1*/)
{
#ifndef _MULTITHREAD    //Call to the block version
  phaseSaitoX_blockZ(V,sdt_x,isMultiregion,isToric,0,V.sizeZ());
#else
  pthread_t * threads = new pthread_t[NbThreads];
  thread_data * thread_data_array = new thread_data[NbThreads];
  int prevz = 0;
  int rc,t;
  pthread_attr_t attr;

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


  //Thread create
  for ( t=0; t<NbThreads - 1; t++)
    {
      thread_data_array[t].thread_id = t;
      thread_data_array[t].V = &V;
      thread_data_array[t].sdt_x = &sdt_x;
      thread_data_array[t].isToric = isToric;
      thread_data_array[t].isMultiregion = isMultiregion;
      thread_data_array[t].minIter = prevz;
      thread_data_array[t].maxIter = prevz + (V.sizeZ() / NbThreads);
      prevz = thread_data_array[t].maxIter ;
      rc = pthread_create(&threads[t], &attr, phaseSaitoX_blockZ_multithread, (void *) &thread_data_array[t]);
    }

  //Last one
  thread_data_array[t].thread_id = t;
  thread_data_array[t].V = &V;
  thread_data_array[t].sdt_x = &sdt_x;
  thread_data_array[t].isToric = isToric;
  thread_data_array[t].isMultiregion = isMultiregion;
  thread_data_array[t].minIter = prevz;
  thread_data_array[t].maxIter = V.sizeZ();
  rc = pthread_create(&threads[t], &attr, phaseSaitoX_blockZ_multithread, (void *) &thread_data_array[t]);

  //Join
  pthread_attr_destroy(&attr);
  for (t = 0; t < NbThreads; t++)
    {
      rc = pthread_join(threads[t], NULL);
      if (rc)
        {
	  printf("ERROR; return code from pthread_join() is %d\n", rc);
	  exit(-1);
        }
    }
#endif
}


/***************************************************************************************************************/
/***************************************************************************************************************/
/*                          Phase Y                                                                            */
/***************************************************************************************************************/
/***************************************************************************************************************/

//--------------------------------------------------------------------
inline void phaseSaitoY_1D(const Vol &V, Longvol &sdt_x, Longvol &sdt_xy, 
			   const bool isMultiregion, const bool isToric, 
			   int x,int z, long int *sdt_temp, unsigned int *s,unsigned int *t)
{
  long int min_value;  //sdt_x min value (used for the toric reencoding
  int index;           // break index for the toric reencoding
  unsigned char TORIC_SHIFT;     // {0,1} To manage the +1 or +0  in the iterator in the Toric domain case
  //int w;
  int q;

  //Brut copy if the row (needed on Toric domain and we hope that this will help the cache access)
  //Index scan
  min_value = sdt_x(x,0,z);
  index = 0;
  TORIC_SHIFT = 0;
  if (isToric)
    {
      for (unsigned int id=1; id < (unsigned int)sdt_x.sizeY(); id++)
	if (sdt_x(x,id,z) < min_value )
	  {
	    min_value = sdt_x(x,id,z);
	    index = id;
	  }
    }

  //Reencoding
  for (unsigned int id=0; id < (unsigned int)sdt_x.sizeY(); id++)
    {
      sdt_temp[id] = prod(sdt_x(x,(id+index) %sdt_x.sizeY(),z),
			  sdt_x(x,(id+index) %sdt_x.sizeY(),z));
      //cout <<prod(sdt_x(x,id,z), sdt_x(x,id,z))<<" ";
    }

  if (isToric)
    {
      //We reconnect the parabolas
      sdt_temp[sdt_x.sizeY()] = sdt_temp[0];
      TORIC_SHIFT = 1;
    }

  //Multiregion: We detect border voxels before the minimization
  if (isMultiregion)
    {
      for (int u=1; u < sdt_x.sizeY()-1 + TORIC_SHIFT ; u++)
        {
	  // std::cerr << "u = " << u << std::endl;
	  if (V(x,u,z) != V(x,u+1,z))
            {
	      //Border voxels detected  -> 0
	      sdt_temp[u] = 0;
	      sdt_temp[u+1] = 0;
	      u++;
            }
        }
      //Y border voxels
      sdt_temp[0] = 0;
      sdt_temp[V.sizeY()-1 + TORIC_SHIFT] = 0;

    }

  //Main Call to the lower Enveloppe Computation
  lowerEnveloppeComputation(sdt_temp,sdt_x.sizeY(),TORIC_SHIFT,s,t,q);

  //Backward Scan
  for (int u = sdt_x.sizeY() -1 + TORIC_SHIFT; u >= TORIC_SHIFT; --u)
    {
      // std::cout << "uu = " << u << std::endl;
      sdt_xy(x,(u+index)%sdt_x.sizeY(),z) = F(u,s[q],sdt_temp[s[q]]);
      if (u==(int)t[q])
	q--;
    }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
#ifdef _MULTITHREAD
void *phaseSaitoY_block_multithread(void *threadarg)
{
  struct thread_data *data;
  data = (struct thread_data* ) threadarg;

  long int *sdt_temp;  //Row copy for the toric domain mode
  unsigned int *s;
  unsigned int *t;

  //Local Memory
  s = (unsigned int *)malloc(sizeof(unsigned int)*((data->sdt_x)->sizeY()+1));
  t = (unsigned int *)malloc(sizeof(unsigned int)*((data->sdt_x)->sizeY()+1));
  sdt_temp = (long int *)malloc(sizeof(long int)*((data->sdt_x)->sizeY()+1));


  cout << "Thread ("<<data->thread_id<<"): "<< data->minIter<<" -> "<<data->maxIter<<endl;

  for (int z = data->minIter; z < data->maxIter ; z++)
    for (int x = 0; x < data->V->sizeX() ; x++)
      {
	phaseSaitoY_1D(*(data->V),*(data->sdt_x),*(data->output),data->isMultiregion,data->isToric,x,z,sdt_temp,s,t);
      }
  free(s);
  free(t);
  free(sdt_temp);
  return 0;
}
#endif

//--------------------------------------------------------------------
void phaseSaitoY_block(const Vol &V, Longvol &sdt_x, Longvol &sdt_xy, 
		       const bool isMultiregion, const bool isToric, 
		       int minZ,int maxZ)
{
  long int *sdt_temp;  //Row copy for the toric domain mode
  unsigned int *s;
  unsigned int *t;

  // std::cerr << "phaseSaitoY_block " << std::endl;

  //Local Memory
  s = (unsigned int *)malloc(sizeof(unsigned int)*(sdt_x.sizeY()+1));
  t = (unsigned int *)malloc(sizeof(unsigned int)*(sdt_x.sizeY()+1));
  sdt_temp = (long int *)malloc(sizeof(long int)*(sdt_x.sizeY()+1));

  for (int z = minZ; z < maxZ ; z++) {
    // std::cerr << "z = " << z << std::endl;
    for (int x = 0; x < V.sizeX() ; x++) {
      phaseSaitoY_1D(V, sdt_x, sdt_xy, isMultiregion, isToric, x, z, sdt_temp, s, t);
    }
  }
  
  free(s);
  free(t);
  free(sdt_temp);
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/**
**************************************************
* @b phaseSaitoY
* @param sdt_x the SDT along the x-direction
* @param sdt_xy the SDT in the xy-slices
**************************************************/
//Second      Step   of    the       saito   algorithm    using    the
//[Meijster/Roerdnik/Hesselink] optimization
void phaseSaitoY(const Vol &V, Longvol &sdt_x, Longvol &sdt_xy, 
		 const bool isMultiregion, const bool isToric, 
		 const int /*NbThreads*/)
{

#ifndef _MULTITHREAD
  // std::cout << "Je *ne* suis *pas* _MULTITHREAD" << std::endl;
  phaseSaitoY_block(V, sdt_x, sdt_xy, isMultiregion, isToric, 0, V.sizeZ());
  //DS : I change V.sizeY() by V.sizeZ()
#else
  // std::cout << "Je  suis  _MULTITHREAD" << std::endl;
  pthread_t * threads = new pthread_t[NbThreads];
  thread_data * thread_data_array = new thread_data[NbThreads];
  int prevz = 0;
  int rc,t;
  pthread_attr_t attr;

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  //Thread create
  for ( t=0; t<NbThreads - 1; t++)
    {
      thread_data_array[t].thread_id = t;
      thread_data_array[t].V = &V;
      thread_data_array[t].sdt_x = &sdt_x;
      thread_data_array[t].output = &sdt_xy;
      thread_data_array[t].isToric = isToric;
      thread_data_array[t].isMultiregion = isMultiregion;
      thread_data_array[t].minIter = prevz;
      thread_data_array[t].maxIter = prevz + (V.sizeZ() / NbThreads);
      prevz = thread_data_array[t].maxIter ;
      rc = pthread_create(&threads[t], &attr, phaseSaitoY_block_multithread, (void *) &thread_data_array[t]);
    }

  //Last one
  thread_data_array[t].thread_id = t;
  thread_data_array[t].V = &V;
  thread_data_array[t].sdt_x = &sdt_x;
  thread_data_array[t].output = &sdt_xy;
  thread_data_array[t].isToric = isToric;
  thread_data_array[t].isMultiregion = isMultiregion;
  thread_data_array[t].minIter = prevz;
  thread_data_array[t].maxIter = V.sizeZ();
  rc = pthread_create(&threads[t], &attr,  phaseSaitoY_block_multithread, (void *) &thread_data_array[t]);

  //Join
  pthread_attr_destroy(&attr);
  for (t = 0; t < NbThreads; t++)
    {
      rc = pthread_join(threads[t], NULL);
      if (rc)
        {
	  printf("ERROR; return code from pthread_join() is %d\n", rc);
	  exit(-1);
        }
    }
#endif
}

/***************************************************************************************************************/
/***************************************************************************************************************/
/*                          Phase Z                                                                            */
/***************************************************************************************************************/
/***************************************************************************************************************/

inline void phaseSaitoZ_1D(const Vol &V, Longvol &sdt_xy, Longvol &sdt_xyz, const bool isMultiregion, const bool isToric, int x,int y, long int *sdt_temp, unsigned int *s,unsigned int *t)
{
  long int min_value;  //sdt_x min value (used for the toric reencoding
  int index;           // break index for the toric reencoding
  unsigned char TORIC_SHIFT;     // {0,1} To manage the +1 or +0  in the iterator in the Toric domain case
  //int w;
  int q;

  //Brut copy if the row (needed on Toric domain and we hope that this will help the cache access)
  //Index scan
  min_value = sdt_xy(x,y,0);
  index = 0;
  TORIC_SHIFT = 0;
  if (isToric)
    {
      for (unsigned int id=1; id < (unsigned int)sdt_xy.sizeZ(); id++)
	if (sdt_xy(x,y,id) < min_value )
	  {
	    min_value = sdt_xy(x,y,id);
	    index = id;
	  }
    }

  //Recoded
  for (unsigned int id=0; id < (unsigned int)sdt_xy.sizeZ(); id++)
    sdt_temp[id] = sdt_xy(x,y,(id+index) %sdt_xy.sizeZ());

  if (isToric)
    {
      //We reconnect the parabolas
      sdt_temp[sdt_xy.sizeZ()] = sdt_temp[0];
      TORIC_SHIFT = 1;
    }

  //We detect border voxels before the min computation
  if (isMultiregion)
    {
      for (int u=1; u < sdt_xy.sizeZ()-1 ; u++)
        {
	  if (V(x,y,u) != V(x,y,u+1))
            {
	      //Border voxels detected
	      sdt_temp[u] = 0;
	      sdt_temp[u+1] = 0;
	      u++;
            }
        }
      // border voxels
      sdt_temp[0] = 0;
      sdt_temp[sdt_xy.sizeZ()-1 + TORIC_SHIFT] = 0;
    }



  lowerEnveloppeComputation(sdt_temp,sdt_xy.sizeZ(),TORIC_SHIFT,s,t,q);


  //Backward Scan
  for (int u = sdt_xy.sizeZ()-1 + TORIC_SHIFT ; u >= TORIC_SHIFT; --u)
    {
      sdt_xyz(x,y,(u+index)%sdt_xy.sizeZ()) = F(u,s[q],sdt_temp[s[q]]);
      if (u== (int)t[q])
	q--;
    }
}

#ifdef _MULTITHREAD
void *phaseSaitoZ_block_multithread(void *threadarg)
{
  struct thread_data *data;
  data = (struct thread_data* ) threadarg;

  long int *sdt_temp;  //Row copy for the toric domain mode
  unsigned int *s;
  unsigned int *t;

  //Local Memory
  s = (unsigned int *)malloc(sizeof(unsigned int)*((data->sdt_x)->sizeZ()+1));
  t = (unsigned int *)malloc(sizeof(unsigned int)*((data->sdt_x)->sizeZ()+1));
  sdt_temp = (long int *)malloc(sizeof(long int)*((data->sdt_x)->sizeZ()+1));


  cout << "Thread ("<<data->thread_id<<"): "<< data->minIter<<" -> "<<data->maxIter<<endl;

  for (int y = data->minIter; y < data->maxIter ; y++)
    for (int x = 0; x < data->V->sizeX() ; x++)
      {
	phaseSaitoZ_1D(*(data->V),*(data->sdt_x),*(data->output),data->isMultiregion,data->isToric,x,y,sdt_temp,s,t);
      }
  free(s);
  free(t);
  free(sdt_temp);
  return 0;
}
#endif
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void phaseSaitoZ_block(const Vol &V, Longvol &sdt_xy, Longvol &sdt_xyz, const bool isMultiregion, const bool isToric, int minY,int maxY)
{
  long int *sdt_temp;  //Row copy for the toric domain mode
  unsigned int *s;
  unsigned int *t;

  //Local Memory
  s = (unsigned int *)malloc(sizeof(unsigned int)*(sdt_xy.sizeZ()+1));
  t = (unsigned int *)malloc(sizeof(unsigned int)*(sdt_xy.sizeZ()+1));
  sdt_temp = (long int *)malloc(sizeof(long int)*(sdt_xy.sizeZ()+1));

  for (int y = minY; y < maxY ; y++) {
    // std::cout << "y=" << y << std::endl;
    for (int x = 0; x < V.sizeX() ; x++)
      {
	phaseSaitoZ_1D(V,sdt_xy,sdt_xyz,isMultiregion,isToric,x,y,sdt_temp,s,t);
	//DS : I change phaseSaitoY_1D by phaseSaitoZ_1D
      }
  }

  free(s);
  free(t);
  free(sdt_temp);
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/**
**************************************************
* @b phaseSaitoZ
* @param sdt_xy the SDT in the xy-slices
* @param sdt_xyz the final SDT
**************************************************/
//Third   Step      of     the    saito   algorithm     using      the
//[Meijster/Roerdnik/Hesselink] optimization
void phaseSaitoZ(const Vol &V, Longvol &sdt_xy, Longvol &sdt_xyz, 
                 const bool isMultiregion, const bool isToric, 
                 const int /*NbThreads*/)
{

#ifndef _MULTITHREAD
  // std::cout << "phaseSaitoZ no _MULTITHREAD " << std::endl;
  phaseSaitoZ_block(V, sdt_xy, sdt_xyz, isMultiregion, isToric, 0, V.sizeY());
  //DS : I change V.sizeZ() by V.sizeY()
#else
  pthread_t * threads = new pthread_t[NbThreads];
  thread_data * thread_data_array = new thread_data [NbThreads];
  int prevz = 0;
  int rc,t;
  pthread_attr_t attr;

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  //Thread create
  for ( t=0; t<NbThreads - 1; t++)
    {
      thread_data_array[t].thread_id = t;
      thread_data_array[t].V = &V;
      thread_data_array[t].sdt_x = &sdt_xy;
      thread_data_array[t].output = &sdt_xyz;
      thread_data_array[t].isToric = isToric;
      thread_data_array[t].isMultiregion = isMultiregion;
      thread_data_array[t].minIter = prevz;
      thread_data_array[t].maxIter = prevz + (V.sizeY() / NbThreads);
      prevz = thread_data_array[t].maxIter ;
      rc = pthread_create(&threads[t], &attr, phaseSaitoZ_block_multithread, (void *) &thread_data_array[t]);
    }

  //Last one
  thread_data_array[t].thread_id = t;
  thread_data_array[t].V = &V;
  thread_data_array[t].sdt_x = &sdt_xy;
  thread_data_array[t].output = &sdt_xyz;
  thread_data_array[t].isToric = isToric;
  thread_data_array[t].isMultiregion = isMultiregion;
  thread_data_array[t].minIter = prevz;
  thread_data_array[t].maxIter = V.sizeY();
  rc = pthread_create(&threads[t], &attr,  phaseSaitoZ_block_multithread, (void *) &thread_data_array[t]);

  //Join
  pthread_attr_destroy(&attr);
  for (t = 0; t < NbThreads; t++)
    {
      rc = pthread_join(threads[t], NULL);
      if (rc)
        {
	  printf("ERROR; return code from pthread_join() is %d\n", rc);
	  exit(-1);
        }
    }
#endif
}

//*******************************************************************************
//*******************************************************************************


///
/// @brief  SEDT computation of the input Vol structure using the Saito's algorithm
/// @param  input the inut vol structure
/// @param  output SEDT result
/// @param  BP the border policy
/// @param  nbThreads the number of threads alloted in the computation
/// @return true in case of success
///
bool computeSEDT(const Vol &input, Longvol &output, const bool isMultiregion, 
		 const bool isToric, const unsigned int NbThreads)
{
  if (!output.isOK())
    {
      fprintf( stderr, "Couldn't init the longvol structure !\n" );
      return false;
    }

  //Creating the intermediate longvol structure
  Longvol sdt_x(output);
  if (!sdt_x.isOK())
    {
      fprintf( stderr, "Couldn't init the longvol structure !\n" );
      return false;
    }

  //  std::cout << "Scan X..." << std::flush; 
  phaseSaitoX(input, output, isMultiregion, isToric, NbThreads);
  // std::cout << " Ok" << std::endl;
  
  // std::cout <<"Scan Y..." << std::flush;
  phaseSaitoY(input, output, sdt_x, isMultiregion, isToric, NbThreads);
  // std::cout << " Ok" << std::endl;

  // std::cout <<"Scan Z..." << std::flush;
  phaseSaitoZ(input, sdt_x, output, isMultiregion, isToric, NbThreads);
  // std::cout << " Ok" << std::endl;

  return true;
}
/**
   =================================================
   * @file   sedt.cc
   * @author David COEURJOLLY <David Coeurjolly <dcoeurjo@liris.cnrs.fr>>
   * @date   Wed Sep 29 17:05:31 2004
   * @date   Fri Jul 4 2008
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
