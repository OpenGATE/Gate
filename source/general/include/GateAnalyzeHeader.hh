/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \brief Class GateAnalyzeHeader :
  \brief Handler for "Analyze" header files (free 3D medical images headers). 
  \brief By laurent.guigues@creatis.insa-lyon.fr

*/

#ifndef __GateAnalyzeHeader_h__
#define __GateAnalyzeHeader_h__

//-----------------------------------------------------------------------------
// namespace	: -
// classe		: AnalyzeHeader
// Handles Analyze Headers (.hdr) for 3D medical images 
// See :
// http://www.mayo.edu/bir/PDF/ANALYZE75.pdf
// http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "globals.hh"
#include "G4ThreeVector.hh"

class G4String;

/// \brief Handler for "Analyze" header files (free 3D medical images
/// headers) 
class GateAnalyzeHeader 
{
public:
  typedef float PixelType;
  //-----------------------------------------------------------------------------
  /// Structure storing general information 
  typedef struct Header_key          		/*      header_key       */
  {                                  		/* off + size*/
    int sizeof_hdr;                		/* 0 + 4     */
    char data_type[10];            		/* 4 + 10    */
    char db_name[18];              		/* 14 + 18   */
    int extents;                   		/* 32 + 4    */
    short int session_error;       		/* 36 + 2    */
    char regular;                  		/* 38 + 1    */
    char hkey_un0;               		/* 39 + 1    */
  } h_k;          				/* total=40  */
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Structure storing the image characteristics : 
  /// dimensions, voxel size, type, etc. 
  typedef struct Image_characteristics  /*      image_dimension  */
  {                               	/* off + size*/
    /* 
       dim[] specifies the image dimensions
       dim[1] - size in x dimension
       dim[2] - size in y dimension
       dim[3] - size in z dimension
       ...
     */
    short int dim[8];           	/* 0 + 16    */
    char vox_units[4];			/* 16 + 4    */
    char cal_units[8];			/* 20 + 4    */
    // I use unused1 to code the number of channels
    short int unused1;			/* 24 + 2    */ 
    // datatype encodes the voxel type (use the constants below)
    short int datatype;			/* 30 + 2    */
    // number of bits per pixel 
    short int bitpix;                   /* 32 + 2    */
    short int dim_un0;                  /* 34 + 2    */
    /* 
       pixdim[] specifies the voxel dimensions:
       pixdim[1] - voxel width
       pixdim[2] - voxel height
       pixdim[3] - interslice distance
       ..etc
    */
    PixelType pixdim[8];                /* 36 + 32   */
    PixelType vox_offset;               /* 68 + 4    */
    PixelType funused1;                 /* 72 + 4    */
    PixelType funused2;                 /* 76 + 4    */
    PixelType funused3;                 /* 80 + 4    */
    PixelType cal_max;                  /* 84 + 4    */
    PixelType cal_min;                  /* 88 + 4    */
    int compressed;                     /* 92 + 4    */
    int verified;                     	/* 96 + 4    */
    int glmax, glmin;                 	/* 100 + 8   */
  } i_c;          			/* total=108 */
  //-----------------------------------------------------------------------------
         
  //-----------------------------------------------------------------------------
  /// Structure storing data concerning the history 
  /// of the acquisition (metadata)
  typedef struct History          	/*      data_history     */
  {                                    	/* off + size*/
    char descrip[80];                	/* 0 + 80    */
    char aux_file[24];               	/* 80 + 24   */
    char orient;                     	/* 104 + 1   */
    char originator[10];             	/* 105 + 10  */
    char generated[10];              	/* 115 + 10  */
    char scannum[10];                	/* 125 + 10  */
    char patient_id[10];             	/* 135 + 10  */
    char exp_date[10];               	/* 145 + 10  */
    char exp_time[10];               	/* 155 + 10  */
    char hist_un0[3];                	/* 165 + 3   */
    int views;                       	/* 168 + 4   */
    int vols_added;                  	/* 172 + 4   */
    int start_field;                 	/* 176 + 4   */
    int field_skip;                  	/* 180 + 4   */
    int omax,omin;                   	/* 184 + 8   */
    int smax,smin;                   	/* 192 + 8   */
  } h;                     		/* total=200 */
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Full header structure
  // typedef
  struct Analyze_struct              		/*      dsr              */
  {                                  		/* off + size*/
    struct Header_key hk;          		/* 0 + 40    */
    struct Image_characteristics ic;   		/* 40 + 108  */
    struct History hist;      		        /* 148 + 200 */
  };						/* total = 348 */
  //-----------------------------------------------------------------------------



  //-----------------------------------------------------------------------------
  /// Type for coding image pixels type
  typedef unsigned char TypeCode;
  /// Code for unknown pixel type
  static const TypeCode UnknownType;
  /// Code for binary pixel type (1 bi)t
  static const TypeCode BinaryType;
  /// Code for unsigned char pixel type
  static const TypeCode UnsignedCharType;
  /// Code for signed short pixel type
  static const TypeCode SignedShortType;
  /// Code for signed int pixel type
  static const TypeCode SignedIntType;
  /// Code for PixelType pixel type
  static const TypeCode FloatType;
  /// Code for complex pixel type
  static const TypeCode ComplexType;
  /// Code for double pixel type
  static const TypeCode DoubleType;
  /// Code for RGB pixel type
  static const TypeCode RGBType;
  /// Code for All pixel type (Meaning ?)
  static const TypeCode AllType;
  //-----------------------------------------------------------------------------



  //-----------------------------------------------------------------------------
  /// Ctor. Creates a default header
  GateAnalyzeHeader();
  /// Dtor
  ~GateAnalyzeHeader() {};

  /// Reads a .hdr file
  /// \param 	filename
  /// \return true if read successful 
  bool Read( const G4String& filename );
  /// Writes the header 
  /// \param 	filename
  /// \return true if read successful 
  bool Write( const G4String& filename );

  /// Returns true iff the header was encoded with the same machine type (little/big endian).

  /// If returns false (after header read) then the data (.img) should be read with endian bytes coding reversal (assuming header and data were encoded the same way).
  bool IsRightEndian() const { return m_rightEndian; }

  /// Returns a const ref on the header information 
  const Analyze_struct& GetData() const { return m_data; }
  /// Returns a ref on the header information 
  Analyze_struct& GetData() { return m_data; }


  /// Sets the type of the voxel
  void SetVoxelType ( TypeCode t ) { 
    m_data.ic.datatype = t; 
    //    m_data.ic.bitpix = BitPix( t ) ;
  }
  /// Gets the type of the voxel
  TypeCode GetVoxelType() { return m_data.ic.datatype; }


  /// Sets the image size
  void SetImageSize ( short int sx, short int sy, short int sz, short int nbchannels = 1) { 
    m_data.ic.dim[1] = sx;
    m_data.ic.dim[2] = sy;
    m_data.ic.dim[3] = sz;
    m_data.ic.unused1 = nbchannels;
  }
  /// Gets the image size
  void GetImageSize ( short int& sx, short int& sy, short int& sz, short int& nbchannels ) { 
    sx = m_data.ic.dim[1];
    sy = m_data.ic.dim[2];
    sz = m_data.ic.dim[3];
    nbchannels = m_data.ic.unused1;
  }
  /// Sets the voxels size 
  //FIXME pixel type is not the same as pixel size type !!!
  void SetVoxelSize ( PixelType sx, PixelType sy, PixelType sz ) { 
    m_data.ic.pixdim[1] = sx;
    m_data.ic.pixdim[2] = sy;
    m_data.ic.pixdim[3] = sz;
  }
  /// Gets the voxels size
  void GetVoxelSize ( PixelType& sx, PixelType& sy, PixelType& sz ) { 
    sx = m_data.ic.pixdim[1];
    sy = m_data.ic.pixdim[2];
    sz = m_data.ic.pixdim[3];
  }
  /// Sets the voxel unit
  void SetVoxelUnit ( const G4String& unit) 
  {
      strcpy(m_data.ic.vox_units,unit);
  }
  /// Sets the voxel unit
  G4String GetVoxelUnit () 
  {
      return m_data.ic.vox_units;
  }




  /// Sets a default header (0x0x0 image)
  void SetDefaults();

  /// Sets the header information to the image's characteristics
  //  void SetData( const Image& i );



protected:
  /// the data
  Analyze_struct m_data;
  /// For hdr oriented img reading : is the header coded with the right endian convention (same as the current machine's one) ?
  bool m_rightEndian;

};
// EO GateAnalyzeHeaderHandler




//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
#endif
