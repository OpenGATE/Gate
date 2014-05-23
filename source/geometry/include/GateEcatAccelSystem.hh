/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateEcatAccelSystem_h
#define GateEcatAccelSystem_h 1

#include "globals.hh"
#include "GateVSystem.hh"

class GateClockDependentMessenger;
class GateToSinoAccel;
class GateSinoAccelToEcat7;

/*! \class  GateEcatAccelSystem
  \brief  The GateEcatAccelSystem models the CTI ECAT ACCEL scanner

  - GateEcatAccelSystem - by delphine.lazaro@imed.jussieu.fr

  - A GateEcatAccelSystem models a scanner that obeys to the CTI ECAT ACCEL family
  . It comprises a hierarchy of components ranging from the
  detector block (detector panels, arranged in several rings) and the crystals.

  - This system overloads two methods of the GateVSystem base-class, Describe() and
  PrintToStream()

  - Beside the standard system methods, it also provides the method ComputeInternalRadius()
  to compute the internal radius of the scanner
*/
class GateEcatAccelSystem : public GateVSystem
{
public:
  GateEcatAccelSystem(const G4String& itsName);  //! Constructor
  virtual ~GateEcatAccelSystem();                //! Destructor

  //! Return the sinogram-maker
  GateToSinoAccel* GetSinogramMaker() const
  { return m_gateToSinoAccel;}

  /*! \brief Method overloading the base-class virtual method Describe().
    \brief This methods prints-out a description of the system, which is
    \brief optimised for creating ECAT7 header files

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void Describe(size_t indent=0);

  /*! \brief Method overloading the base-class virtual method Describe().
    \brief This methods prints out description of the system to a stream.
    \brief It is essentially to be used by the class GateToSinogram, but it may also be used by Describe()

    \param aStream: the output stream
    \param doPrintNumbers: tells whether we print-out the volume numbers in addition to their dimensions
  */
  virtual void PrintToStream(std::ostream& aStream,G4bool doPrintNumbers);


  //! Compute the internal radius of the crystal ring.
  virtual G4double ComputeInternalRadius();

private:
  GateClockDependentMessenger    	*m_messenger; 	//!< Messenger

  GateToSinoAccel                     *m_gateToSinoAccel;
#ifdef GATE_USE_ECAT7
  GateSinoAccelToEcat7              *m_gateSinoAccelToEcat7;
#endif
};

#endif
