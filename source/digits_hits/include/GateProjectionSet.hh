/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateProjectionSet_H
#define GateProjectionSet_H

#include "globals.hh"
#include <fstream>
#include "GateDetectorConstruction.hh"

/*!
  \file GateProjectionSet.hh

  $Log: GateProjectionSet.hh,v $
  Revision 1.1.1.1.4.1  2011/03/10 16:32:35  henri
  Implemented multiple energy window interfile output

  Revision 1.1.1.1.4.1  2011/02/02 15:37:46  henri
  Added support for multiple energy windows

  Revision 1.6  2010/11/30 16:33:22  henri
  bug correction

  Revision 1.5  2010/11/26 14:26:39  henri
  *** empty log message ***


  Revision 1.2
  date: 2010-11-26 14:35:15 +0100;
Class GateProjectionSet

Changes for adding multiple energy window support. The projection data is now a 3D array, storing each energy window and each head for 1 camera position.

Added attribute :
	size_t m_energyWindowNb to store the total of energy windows

Modified attributes :
	ProjectionDataType** m_data -> ProjectionDataType*** m_data (was [head][pixel], now is [energyWindow][head][pixel]
	ProjectionDataType* m_dataMax -> ProjectionDataType** m_dataMax (was [head], now is [energyWindow][head])

Methods added :
	inline size_t GetEnergyWindowNb() const
	inline void SetEnergyWindowNb(size_t)
	inline G4int BytesPerEnergyWindow() const

Methods modified with no prototype change:
	ClearData(size_t) : Clearing of 3D array instead of 3D array

Methods modified which prototype changed :
	Reset(size_t headNb, size_t projectionNb) -> Reset(size_t energyWindowNb, size_t headNb, size_t projectionNb) : Allocation of 3D array regarding to the new m_data and m_dataMax type
	Fill(G4int, G4double x, G4double y) -> Fill(G4int energyWindowID, G4int headID, G4double, G4double)
	StreamOut(std::ofstream&, size_t headID) -> StreamOut(std::ofstream&, size_t energyWindowID, size_t headID)

Methods overloaded :
	GetMaxCounts(size_t headID) -> Added GetMaxCounts(size_t energyWindowID, headID) : Get the maximum for 1 energy window and 1 head

Modified some getters and setters to work with the new implementation :
	inline ProjectionDataType* GetProjection(size_t headID) const -> inline ProjectionDataType* GetProjection(size_t energyWindowID, size_t headID) const
	inline ProjectionDataType** GetData() const -> inline ProjectionDataType*** GetData() const


*/



/*! \class  GateProjectionSet
    \brief  Structure to store the projection sets from a SPECT simulation

    - GateProjectionSet - by Daniel.Strul@iphe.unil.ch (Oct. 2002)

    - This structure is generated during a SPECT simulation by GateToProjectionSet. It can be stored
      into an output file using a set-writer such as GateToInterfile

    \sa GateToProjectionSet, GateToInterfile
*/


/* Modified by HDS on 25/11/2010 <henri.dersarkissian@gmail.com>
	Adding multiple energy window support for SPECT
	ProjectionSet now record more than 1 energy window, and still multiple heads
*/

class GateProjectionSet
{
  public:
    typedef unsigned short ProjectionDataType;
    typedef G4double ARFProjectionDataType;
  public:

    inline GateProjectionSet();       	      	      	  //!< Public constructor
    inline virtual ~GateProjectionSet() {Reset();} 	  //!< Public destructor

    //! Reset the matrix and prepare a new acquisition
    // Modified by HDS : Added size_t energyWindowNb=0 parameter
    void Reset(size_t energyWindowNumber=0,size_t headNumber=0,size_t projectionNumber=0);

    //! Clear the matrix and prepare a new run
    void ClearData(size_t projectionID);

    //! Store a digi into a projection
    void Fill( G4int energyWindowID, G4int headID, G4double x, G4double y);
    void FillARF( G4int, G4double , G4double , G4double); /*PY Descourt 08/09/2009*/
    //! \name getters and setters
    //@{

	//! Returns the number of energy windows
    inline size_t GetEnergyWindowNb() const
      { return m_energyWindowNb;}
    //! Set the number of energy windows
    inline void SetEnergyWindowNb(size_t aNb)
      { m_energyWindowNb = aNb;}

    //! Returns the number of heads
    inline size_t GetHeadNb() const
      { return m_headNb;}
    //! Set the number of heads
    inline void SetHeadNb(size_t aNb)
      { m_headNb = aNb;}

    //! Returns the number of projections per head
    inline size_t GetProjectionNb() const
      { return m_projectionNb;}
    //! Set the number of projections per head
    inline void SetProjectionNb(size_t aNb)
      { m_projectionNb = aNb;}

    //! Returns the number of pixels along X
    inline G4int GetPixelNbX() const
      { return m_pixelNbX;}
    //! Set the number of pixels along X
    inline void SetPixelNbX(G4int aNb)
      { m_pixelNbX = aNb; ComputeLowEdges(); }

     //! Returns the number of pixels along Y
    inline G4int GetPixelNbY() const
      { return m_pixelNbY;}
    //! Set the number of pixels along Y
    inline void SetPixelNbY(G4int aNb)
      { m_pixelNbY = aNb; ComputeLowEdges(); }

    //! Returns the pixel size along X
    inline G4double GetPixelSizeX() const
      { return m_pixelSizeX;}
    //! Set the pixel size along X
    inline void SetPixelSizeX(G4double aSize)
      { m_pixelSizeX = aSize; ComputeLowEdges(); }

     //! Returns the pixel size along Y
    inline G4double GetPixelSizeY() const
      { return m_pixelSizeY;}
    //! Set the pixel size along Y
    inline void SetPixelSizeY(G4double aSize)
      { m_pixelSizeY = aSize; ComputeLowEdges(); }

    //! Returns the data pointer
    inline ProjectionDataType*** GetData() const
      { return m_data;}

    //! Set the verbose level
    virtual void SetVerboseLevel(G4int val)
      { m_verboseLevel = val; };

    //! Returns a projection from the set
    // Modified by HDS : need to specify the energy window ID for multiple energy windows
    inline ProjectionDataType* GetProjection(size_t energyWindowID, size_t headID) const
      { return m_data[energyWindowID][headID];}

    //! Returns the number of pixels per projection
    inline G4int PixelsPerProjection() const
      { return m_pixelNbX * m_pixelNbY;}

     //! Returns the number of bytes per projection
    inline G4int BytesPerProjection() const
      { return PixelsPerProjection() * BytesPerPixel() ;}

     //! Returns the number of bytes per head
    inline G4int BytesPerHead() const
      { return BytesPerProjection() * m_projectionNb ;}

    //! Returns the number of bytes per energy window
    // Added by HDS : multiple energy windows
    inline G4int BytesPerEnergyWindow() const
      { return BytesPerHead() * m_headNb ;}

     //! Returns the nb of bytes per pixel
    inline size_t BytesPerPixel() const
      { return sizeof(ProjectionDataType);}

	 // Modified by HDS : For multple energy window support. This function only works for ARF data,
	 // otherwise one must use the overloaded function : GetMaxCounts(energyWindowID, headID)
     //! Returns the data-max counter for a head
    inline ProjectionDataType GetMaxCounts(size_t headID) const /*PY Descourt 08/09/2009*/
      {
      	if ( m_ARFdataMax !=0 ) return ((ProjectionDataType)m_ARFdataMax[headID]);
      	return 0;
	  }


      // Added by HDS for multiple energy windows support
      //! Returns the max data for an energy window for one head
      inline ProjectionDataType GetMaxCounts(size_t energyWindowID, size_t headID) const
      {
      	if ( m_dataMax !=0 ) return m_dataMax[energyWindowID][headID];
      	if ( m_ARFdataMax !=0 ) return ((ProjectionDataType)m_ARFdataMax[headID]);
      	return 0;
      }

     //! Returns the data-max counter for a head
    inline size_t GetCurrentProjectionID() const
      { return m_currentProjectionID;}

/*PY Descourt 08/09/2009*/
    //! Returns the ARF data pointer
    inline ARFProjectionDataType** GetARFData() const
      { return m_ARFdata;}
	      //! Returns a projection from the set
    inline ARFProjectionDataType* GetARFProjection(size_t headID) const
      { return m_ARFdata[headID];}


     //! Returns the number of bytes per projection
    inline G4int ARFBytesPerProjection() const
      { return PixelsPerProjection() * ARFBytesPerPixel() ;}

     //! Returns the number of bytes per head
    inline G4int ARFBytesPerHead() const
      { return ARFBytesPerProjection() * m_projectionNb ;}

     //! Returns the nb of bytes per pixel
    inline size_t ARFBytesPerPixel() const
      { return sizeof(ARFProjectionDataType);}
	  /*PY Descourt 08/09/2009*/

  //@}

    //! Compute the low edge of the matrix
    inline void ComputeLowEdges()
      {
      	m_matrixLowEdgeX = -0.5 * m_pixelNbX * m_pixelSizeX ;
      	m_matrixLowEdgeY = -0.5 * m_pixelNbY * m_pixelSizeY ;
      }


    /*! \brief Writes a head-projection onto an output stream

      	\param dest:    	  the destination stream
      	\param energyWindowID:	the energy window whose projection to stream-out
      	\param headID:    	  the head whose projection to stream-out
    */
    void StreamOut(std::ofstream& dest, size_t energyWindowID, size_t headID);
    void StreamOutARFProjection(std::ofstream&, size_t);/*PY Descourt 08/09/2009*/
    //! \name Data fields
    //@{

	size_t				  m_energyWindowNb;					//!< Nb of energy windows
    size_t    	      	  m_headNb;     	      	      	//!< Nb of scanner heads
    size_t    	      	  m_projectionNb;     	      	      	//!< Nb of projections per head
    G4int     	      	  m_pixelNbX,m_pixelNbY; 	      	//!< Nb of pixels per projection along X and Y
    G4double  	      	  m_pixelSizeX,m_pixelSizeY; 	      	//!< Pixel sizes along X and Y
    G4double  	      	  m_matrixLowEdgeX,m_matrixLowEdgeY;    //!< Low edge of the matrix (-n*dx/2)
    ProjectionDataType ***m_data;       	      	      	//!< Array of data sets
    ProjectionDataType  **m_dataMax;       	      	      	//!< Max count for each projection
    G4int     	      	  m_currentProjectionID;	      	//!< ID of the current projection
    G4int     	      	  m_verboseLevel;

    ARFProjectionDataType **m_ARFdata;/*PY Descourt 08/09/2009*/
    ARFProjectionDataType *m_ARFdataMax;/*PY Descourt 08/09/2009*/
	long unsigned int     m_rec;/*PY Descourt 08/09/2009*/
    long unsigned int     m_rej;/*PY Descourt 08/09/2009*/
    //@}

};




// Public constructor
inline GateProjectionSet::GateProjectionSet()
  : m_energyWindowNb(0)
  , m_headNb(0)
  , m_projectionNb(0)
  , m_pixelNbX(0),m_pixelNbY(0)
  , m_pixelSizeX(0.),m_pixelSizeY(0.)
  , m_matrixLowEdgeX(0.),m_matrixLowEdgeY(0.)
  , m_data(0)
  , m_dataMax(0)
  , m_currentProjectionID(-1)
  , m_verboseLevel(0)
{
  m_ARFdata = 0;
  m_ARFdataMax = 0;
}


#endif
