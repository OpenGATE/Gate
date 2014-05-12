/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateOscTranslationMove_h
#define GateOscTranslationMove_h 1

#include "globals.hh"

#include "GateVGlobalPlacement.hh"

class GateOscTranslationMoveMessenger;

/*! \class  GateOscTranslationMove
    \brief  The GateOscTranslationMove models an oscillating translation movement
    
    - GateOscTranslationMove - by Daniel.Strul@iphe.unil.ch (Aug. 10, 2002)
    
    - The movement equation modeled by this class is dM(t) = A x sin (2 PI f t + phi) 
      where dM(t) is the translation vector at time t, A is the maximum displacement vector,
      f is the movement frequency, phi is the phase a t=0, and t is the time
      
*/      
class GateOscTranslationMove  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default movement parameters are chosen so that, by default, the movement is null
    GateOscTranslationMove(GateVVolume* itsObjectInserter,
      	      	      	      	  const G4String& itsName,
				  const G4ThreeVector& itsAmplitude=G4ThreeVector(),
				  G4double itsFrequency=0.,
				  G4double itsPhase=0.);
    //! Destructor
    virtual ~GateOscTranslationMove();

    /*! \brief Implementation of the pure virtual method PushMyPlacement(), to compute
      	\brief a volume position and orientation as a function of time

	\param currentRotationMatrix: the rotation matrix that defines the current orientation of the volume
	\param currentPosition:       the vector that defines the current position of the volume
	\param aTime:                 the current time
	
	\return an object GatePlacement combining the new position and the new orientation of the volume
	        computed by the movement equation
    */    
    virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	      	      const G4ThreeVector& currentPosition,
			      	      	      G4double aTime);

    /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
      	\brief a description of the movement

	\param indent: the print-out indentation (cosmetic parameter)
    */    
    virtual void DescribeMyself(size_t indent);
     
  public:
    //! \name getters and setters
    //@{
    //! Get the maximum displacement vector
    inline const G4ThreeVector& GetAmplitude()  { return m_amplitude;}
    //! Set the maximum displcament vector
    void SetAmplitude(const G4ThreeVector& val) { m_amplitude = val;  }

    //! Get the oscillation frequency
    inline G4double GetFrequency()   { return m_frequency;}
    //! Set the oscillation frequency
    void SetFrequency(G4double val)  { m_frequency = val;  }

    //! Return the oscillation period, based on the value of the frequency
    inline G4double GetPeriod()  { return m_frequency ? 1./m_frequency : DBL_MAX;}
    //! Set the oscillation period
    void SetPeriod(G4double val) { m_frequency = val ? 1./val : DBL_MAX;  }

    //! Get the oscillation phase a t=0
    inline G4double GetPhase()   { return m_phase;}
    //! Set the oscillation phase a t=0
    void SetPhase(G4double val)  { m_phase = val;  }

    //! Get the value of the translation vector that was last computed by PushMyPlacement()
    inline const G4ThreeVector& GetCurrentTranslation() { return m_currentTranslation;}
    //@}

  private:
    //! \name movement parameters
    //@{
    G4ThreeVector m_amplitude;	  //!< maximum displacement vector
    G4double  	  m_frequency;	  //!< oscillation frequency
    G4double  	  m_phase;    	  //!< oscillation phase a t=0
    //@}

    //! Translation vector that was last computed by PushMyPlacement()
    G4ThreeVector m_currentTranslation;
    
    //! Messenger
    GateOscTranslationMoveMessenger* m_Messenger; 

};

#endif

