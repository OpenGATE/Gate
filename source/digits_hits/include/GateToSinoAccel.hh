/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateToSinoAccel_H
#define GateToSinoAccel_H

#include "fstream"

#include "GateVOutputModule.hh"
#include "GateSinogram.hh"

class GateVSystem;
class GateToSinoAccelMessenger;

class GateToSinoAccel :  public GateVOutputModule
{
public:
  //! Public constructor (creates an empty, uninitialised, project set)
  GateToSinoAccel(const G4String& name, GateOutputMgr* outputMgr,GateVSystem* itsSystem,DigiMode digiMode);

  virtual ~GateToSinoAccel();
  const G4String& GiveNameOfFile();

  //! Initialisation of the projection set
  void RecordBeginOfAcquisition();
  //! We leave the projection set as it is (so that it can be stored afterwards)
  //! but we still have to destroy the array of projection IDs
  void RecordEndOfAcquisition();
  //! Reset the projection data
  void RecordBeginOfRun(const G4Run *);
  //! Write 2D sinograms
  void RecordEndOfRun(const G4Run *);
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

    //! Returns the value of the raw ouptut enabled/disabled status flag
    inline virtual G4bool IsRawOutputEnabled() const
    	  { return m_flagIsRawOutputEnabled;}
    //! Enable the raw output
    inline virtual void RawOutputEnable(G4bool val)
          { m_flagIsRawOutputEnabled = val; }


    //! Returns the value of the trues only status flag
    inline virtual G4bool IsTruesOnly() const
          { return m_flagTruesOnly; }
    //! Enable recording of true coincidences only
    inline virtual void TruesOnly(G4bool val)
          { m_flagTruesOnly = val; }

    //! Get the output file name
    const  G4String& GetFileName()
          { return m_fileName;       };
    //! Set the output file name
    void   SetFileName(const G4String& aName)
          { m_fileName = aName;      };

    //! Get the input data channel name
    const  G4String& GetInputDataName()
          { return m_inputDataChannel;       };
    //! Set the output data channel name
    void   SetOutputDataName(const G4String& aName)
          { m_inputDataChannel = aName;      };


    //! Overload of the base-class' method: we command both our own verbosity and that of the sinogram
    inline void SetVerboseLevel(G4int val)
      { GateVOutputModule::SetVerboseLevel(val); m_sinogram->SetVerboseLevel(val); }


     //! Returns the study duration
    inline G4double GetStudyDuration() const
      { return m_studyDuration;}

    //! Returns the frame duration
    inline G4double GetFrameDuration() const
      { return m_frameDuration;}

    //! Returns thr number of frames
    inline G4int GetFrameNb() const
      { return m_frameNb;}

    //! Returns the 2D sinogram
    inline GateSinogram* GetSinogram() const
      { return m_sinogram;}

    //! Returns the number of crystals per crystal ring
    inline G4int GetCrystalNb() const
      { return m_sinogram->GetCrystalNb();}
    //! Set the number of crystals per crystal ring
    inline void SetCrystalNb(size_t aNb)
      { m_sinogram->SetCrystalNb(aNb);}

     //! Returns the number of crystal rings
    inline G4int GetRingNb() const
      { return m_sinogram->GetRingNb();}
    //! Set the number of crystal rings
    inline void SetRingNb(size_t aNb)
      { m_sinogram->SetRingNb(aNb);}

     //! Returns the number of radial sinogram bins
     inline size_t GetRadialElemNb() const
       { return  m_sinogram->GetRadialElemNb();}
     //! Set the number of radial sinogram bin
     inline void SetRadialElemNb(size_t aNb)
       { m_sinogram->SetRadialElemNb(aNb);}

     //! Returns the FWHM of crystal location resolution in tangential direction
     inline G4double GetTangCrystalResolution() const
       { return m_tangCrystalResolution;}
     //! Set the FWHM of crystal location resolution in tangential direction
     inline void SetTangCrystalResolution(G4double aNb)
       { m_tangCrystalResolution = aNb;}
     //! Returns the FWHM of crystal location resolution in axtial direction
     inline G4double GetAxialCrystalResolution() const
       { return m_axialCrystalResolution;}
     //! Set the FWHM of crystal location resolution in axtial direction
     inline void SetAxialCrystalResolution(G4double aNb)
       { m_axialCrystalResolution = aNb;}

     //! Returns the nb of bytes per pixel;
    inline size_t BytesPerPixel() const
      { return m_sinogram->BytesPerPixel();}

   //@}

protected:
  GateSinogram*       m_sinogram;	      	  //!< 2D sinogram for PET simulations
  // GateSinogram*       m_sinoRandoms;              //!< Additional 2D sinograms for random coincidences
  size_t       	      m_crystalNb;    	      	  //!< Total number of crystals per crystal ring
  size_t      	      m_ringNb;       	      	  //!< Number of crystal rings
  G4double    	      m_studyDuration;	      	  //!< Total duration of the acquisition
  G4double            m_frameDuration;            //!< Total duration of one frame
  G4int               m_frameNb;                  //!< Number of frames
  GateVSystem        *m_system;                   //!< Pointer to the system, used to get the system information
  G4bool              m_flagTruesOnly;            //!< Defines whether randoms are recorded or not
  size_t              m_radialElemNb;             //!< Total number of radial sinogram bins
  G4bool	      m_flagIsRawOutputEnabled;	  //!< Defines whether the raw-file output is active or inactive
  G4String            m_fileName;                 //!< raw output files name
  G4double            m_tangCrystalResolution;    //!< FWHM of crystal location resolution in tangential direction
  G4double            m_axialCrystalResolution;   //!< FWHM of crystal location resolution in axial direction
  GateToSinoAccelMessenger *m_messenger;
  G4String	      m_inputDataChannel;	  //!< Name of the coincidence-collection to store into the sinogram

  // CC & AC: Durty work
  // G4std::ofstream     m_dataFile;   	      	  //!< Output stream for the data file

};

#endif
