/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!

   \file GateToProjectionSet.hh

   $Log: GateToProjectionSet.hh,v $

   Revision v6.2   2012/07/09  by vesna.cuplov@gmail.com
   Implemented functions that have a link with the GateToOpticalRaw class for Optical photons which is
   used to write as an output file the result of the GateToProjectionSet module for Optical Photons.

   Revision 1.1.1.1.4.1  2011/03/10 16:32:35  henri
   Implemented multiple energy window interfile output

   Revision 1.1.1.1.4.1  2011/02/02 15:37:46  henri
   Added support for multiple energy windows

   Revision 1.4  2010/12/01 17:11:23  henri
   Various bug fixes

   Revision 1.3  2010/11/30 17:48:50  henri
   Comments

   Revision 1.3  2010/11/30 16:47:08  henri
   Class GateToProjectionSet
   Modifications in order to record more than 1 input channel, as energy window

   ***Interface Changes***
   Attribute added:
   	- size_t m_energyWindowNb : the number of energy window to record

   Attributes modified:
   	- G4String m_inputDataChannel -> std::vector<G4String> m_inputDataChannelList
   	- G4int m_inputDataChannelID -> std::vector<G4int> m_inputDataChannelIDList

   Getters and setters modified:
   	- const G4String& GetInputDataName()  -> const G4String& GetInputDataName(size_t energyWindowID) const : get the input channel name by index
   	- void   SetOutputDataName(const G4String& aName) -> void  SetInputDataName(const G4String& aName) : changed the name

   Getters and setters added:
   	- inline size_t GetEnergyWindowNb() const
   	- G4int GetInputDataID(size_t energyWindowID) const : Get the input channel ID by index
   	- void  AddInputDataName(const G4String& aName) : Add a input data channel name to the list
   	- std::vector<G4String> GetInputDataNameList() const
   	- std::vector<G4int> GetInputDataIDList() const


   ***Implementation Changes***
   In GateToProjectionSet.hh:
   	- GetTotalImageNb() now returns the number of images for all energy windows

   In GateToProjectionSet.cc:
   	- Slightly changed the constructor
   	- RecordBeginOfAcquisition :
   		*Loop over the input data channel list to get the channel IDs and fill the m_inputDataChannelIDList vector
   		*Initialize the attribute m_energyWindowNb with the size of the vector m_inputDataChannelList
   		*Updated GateProjectionSet::Reset

   	- RecordEndOfEvent :
   		*Changed some verbose text to show which digi chain is recorded (aka energy window)
   		*Loop over all energy windows in m_inputDataChannelList to store all the digis related to this event



*/

#ifndef GateToProjectionSet_H
#define GateToProjectionSet_H

#include <fstream>

#include "GateVOutputModule.hh"
#include "GateProjectionSet.hh"
#include "GateToInterfile.hh"
#include "GateToOpticalRaw.hh" // v. cuplov -- GateToOpticalRaw for optical photons

class GateVSystem;
class GateToProjectionSetMessenger;

class GateToProjectionSet :  public GateVOutputModule
{
public:
  //! Public constructor (creates an empty, uninitialised, project set)
  GateToProjectionSet(const G4String& name, GateOutputMgr* outputMgr,GateVSystem* itsSystem,DigiMode digiMode);

  virtual ~GateToProjectionSet();
  const G4String& GiveNameOfFile();

  // Functions for messenger commands that have a link with the GateToInterfile class
  void SetVerboseToProjectionSetAndInterfile(G4int aVerbosity);
  void SendDescribeToProjectionSetAndInterfile();
  void SetEnableToProjectionSetAndInterfile();
  void SetDisableToProjectionSetAndInterfile();

  //! Initialisation of the projection set
  void RecordBeginOfAcquisition();
  //! We leave the projection set as it is (so that it can be stored afterwards)
  //! but we still have to destroy the array of projection IDs
  void RecordEndOfAcquisition();
  //! Reset the projection data
  void RecordBeginOfRun(const G4Run *);
  //! Nothing to do
  void RecordEndOfRun(const G4Run *) {}
  //! Nothing to do
  void RecordBeginOfEvent(const G4Event *) {}
  //! Update the target projections with regards to the digis acquired for this event
  void RecordEndOfEvent(const G4Event *);
  //! Nothing to do for steps
  void RecordStepWithVolume(const GateVVolume *, const G4Step *) {}
  //! Nothing to do
  void RecordVoxels(GateVGeometryVoxelStore *) {};

    /*! \brief Overload of the base-class' virtual method to print-out a description of the module

	\param indent: the print-out indentation (cosmetic parameter)
    */
    void Describe(size_t indent=0);


    //! \brief Writes the projection sets onto an output stream
    void StreamOut(std::ofstream& dest);

    //! \name getters and setters for the projection maker
    //@{

    //! Overload of the base-class' method: we command both our own verbosity and that of the projection set
    inline void SetVerboseLevel(G4int val)
      { GateVOutputModule::SetVerboseLevel(val); m_projectionSet->SetVerboseLevel(val); }

    //! return the total angular span of the projection set
    inline G4double GetAngularSpan() const
      { return m_orbitingStep * m_projNb;}

    //! return the time per projection
    inline G4double GetTimePerProjection() const
      { return m_studyDuration / m_projNb;}

    //! return the angulart pitch between heads
    inline G4double GetHeadAngularPitch() const
      { return m_headAngularPitch;}


     //! Returns the study duration
    inline G4double GetStudyDuration() const
      { return m_studyDuration;}

    //! Set the sampling plane
    void SetProjectionPlane(const G4String& planeName);

    //! Returns the number of projections per head
    inline size_t GetProjectionNb() const
      { return m_projNb;}

    //! Returns the number of heads per energy window
    inline size_t GetHeadNb() const
      { return m_headNb;}

    //! Returns the number of energy windows
    inline size_t GetEnergyWindowNb() const
      { return m_energyWindowNb;}

    //! Returns the total number of images
    inline size_t GetTotalImageNb() const
      { return GetProjectionNb() * GetHeadNb() * GetEnergyWindowNb();}


	//! Get the input data channel name for energy window
    const G4String& GetInputDataName(size_t energyWindowID) const
          { return m_inputDataChannelList.at(energyWindowID);       };

    //! Get the input data channel id for energy window
    G4int GetInputDataID(size_t energyWindowID) const
          { return m_inputDataChannelIDList.at(energyWindowID);       };

    //! Add the input data channel name
    void   AddInputDataName(const G4String& aName)
          { m_inputDataChannelList.push_back(aName);      };

    //! Get the list of input data channel name
    std::vector<G4String> GetInputDataNameList() const
          { return m_inputDataChannelList;       };

    //! Get the list of input data channel id
    std::vector<G4int> GetInputDataIDList() const
          { return m_inputDataChannelIDList;       };

    //! Reset the list of intput data channel and add a name
    void  SetInputDataName(const G4String& aName)
          { m_inputDataChannelList.clear();
          	AddInputDataName(aName);		};

    //! Set the output file name in the GateToInterfile class
    void   SetOutputFileName(const G4String& aName);

    //@}

    //! \name getters and setters to access the fields of the projection set
    //@{

    //! Returns the projection set
    inline GateProjectionSet* GetProjectionSet() const
      	  { return m_projectionSet;}

    //! Returns the number of pixels along X
    inline G4int GetPixelNbX() const
      { return m_projectionSet->GetPixelNbX();}
    //! Set the number of pixels along X
    inline void SetPixelNbX(G4int aNb)
      { m_projectionSet->SetPixelNbX(aNb);}

     //! Returns the number of pixels along Y
    inline G4int GetPixelNbY() const
      { return m_projectionSet->GetPixelNbY();}
    //! Set the number of pixels along Y
    inline void SetPixelNbY(G4int aNb)
      { m_projectionSet->SetPixelNbY(aNb);}

    //! Returns the pixel size along X
    inline G4double GetPixelSizeX() const
      { return m_projectionSet->GetPixelSizeX();}
    //! Set the pixel size along X
    inline void SetPixelSizeX(G4double aSize)
      { m_projectionSet->SetPixelSizeX(aSize);}

     //! Returns the pixel size along Y
    inline G4double GetPixelSizeY() const
      { return m_projectionSet->GetPixelSizeY();}
    //! Set the pixel size along Y
    inline void SetPixelSizeY(G4double aSize)
      { m_projectionSet->SetPixelSizeY(aSize);}

     //! Returns the nb of bytes per pixel;
    inline size_t BytesPerPixel() const
      { return m_projectionSet->BytesPerPixel();}

   //@}

protected:
  GateProjectionSet*  m_projectionSet;	      	  //!< Projection set for SPECT or OPTICAL simulations
  size_t			  m_energyWindowNb;			//!< Number of energy windows
  size_t       	      m_projNb;    	      	  //!< Total number of projections
  size_t      	      m_headNb;       	      	  //!< Number of heads
  G4double            m_orbitingStep;    	  //!< Angular step between runs
  G4double            m_headAngularPitch;    	  //!< Angular step between heads
  G4String    	      m_projectionPlane;	  //!< The name of the projection plane
  size_t      	      m_coordX; 	      	  //!< The coordinate axis to use for the X coord
  size_t      	      m_coordY; 	      	  //!< The coordinate axis to use for the Y coord
  G4double    	      m_studyDuration;	      	  //!< Total duration of the simulation

  //! Pointer to the system, used to get the system information (rotation speed...)
  GateVSystem *m_system;

  GateToProjectionSetMessenger *m_messenger;

  std::vector<G4String>	      m_inputDataChannelList;	  //!< Name of the coincidence-collection to store into the sinogram
  std::vector<G4int> 	      m_inputDataChannelIDList;
  G4String            m_noFileName;

  GateToInterfile*    m_gateToInterfile;
  GateToOpticalRaw*   m_gateToOpticalRaw; // v. cuplov -- GateToOpticalRaw for optical photons


};

#endif
