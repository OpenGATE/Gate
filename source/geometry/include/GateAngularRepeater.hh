/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateAngularRepeater_h
#define GateAngularRepeater_h 1

#include "globals.hh"
#include "G4SystemOfUnits.hh"

#include "GateVGlobalPlacement.hh"
#include "G4Point3D.hh"
#define ModuloMax 8

class GateVVolume;
class GateAngularRepeaterMessenger;

/*! \class  GateAngularRepeater
  \brief  The GateAngularRepeater models a repetition of an object around an axis,
  \brief  a pattern similar to the repetition of the scanner blocks in a cylindrical PET.
  \brief  A Rsector can be shifted along Z each modulo N Rsector starting at a specified Rsector.
  \brief  Each element of a group of [m_moduloNumber] is shifted according to its position inside a
  sequence (see "j" bellow).
  \brief  Without specifing the shift(s) and the sequence length in the macro sequence, we get no shift and the
  standard ring repeater.
  \brief  In a [m_moduloNumber] sequence, elements shift value is defined as :
  \brief   Z shift(i) = Zshift_j where j=(i%m_moduloNumber)+1 : position in a sequence and i position on the full ring.
  \brief  See also "List Mode Format Implementation: Scanner geometry description Version 4.1 M.Krieguer & al." ,
  \brief  in particular ASCII header file entrie "z shift sector [NIDShifted] mod [Nperiodicity]: [Zshift] [units]"
  \brief  related : GateToLMF class.
  Please take care that the present LMF format (22.10.03) support only for the moment a geometry with
  a cylindrical symetry. For example, a repeater starting at 0 degre and finishing at 90 degree (a quarter of
  ring) wont be supported by the LMF output.


  - The angular repeater uses six parameters:
  - a number of repetitions
  - a start point and end-point of an axis
  - an autorotation flag
  - a start angle and an angular span
  - a repeated sequence length
  - the Z shift of element in the "modulo" sequence.
  Based on these parameters, it repeats an object at regular steps around an axis.
  If the autorotation flag is on, the orientation of the copies changes along with
  their position, as in a PET scanner geometry. If this flag is off, their
  orientation is the same for all copies (like in a fun-fair wheel).
*/
class GateAngularRepeater  : public GateVGlobalPlacement
{
public:
  //! Constructor
  //! The default repeater parameters are chosen so that, by default, the object is unchanged
  GateAngularRepeater(GateVVolume* itsObjectInserter,
                      const G4String& itsName="ringmodulo",
                      G4int itsRepeatNumber=1,
                      const G4Point3D& itsPoint1=G4Point3D(),
                      const G4Point3D& itsPoint2=G4Point3D(0.,0.,1.),
                      G4bool itsFlagAutoRotation=true,
                      G4double itsFirstAngle=0.,
                      G4double itsAngularSpan=360. * deg,
                      G4int itsModuloNumber=1,
                      G4double itsZShift1=0. * mm,
                      G4double itsZShift2=0. * mm,
                      G4double itsZShift3=0. * mm,
                      G4double itsZShift4=0. * mm,
                      G4double itsZShift5=0. * mm,
                      G4double itsZShift6=0. * mm,
                      G4double itsZShift7=0. * mm,
                      G4double itsZShift8=0. * mm);
  //! Destructor
  virtual ~GateAngularRepeater();

public:
  /*! \brief Implementation of the pure virtual method PushMyPlacements(), to compute
    \brief the position and orientation of all copies as a function of time. The series
    \brief of placements thus obtained is placed into the repeater placement queue.

    \param currentRotationMatrix: the rotation matrix that defines the current orientation of the volume
    \param currentPosition:       the vector that defines the current position of the volume
    \param aTime:                 the current time

  */
  virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                const G4ThreeVector& currentPosition,
                                G4double aTime);
  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the repeater

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t indent);

public:
  //! \name getters and setters
  //@{

  //! Get the number of repetitions
  inline G4int GetRepeatNumber() const      	  { return m_repeatNumber;}
  //! Get the starting point of the repetition axis
  inline const G4Point3D& GetPoint1() const 	  { return m_point1 ;}
  //! Get the end point of the repetition axis
  inline const G4Point3D& GetPoint2() const 	  { return m_point2 ;}
  //! Get the value of the auto-rotation flag
  inline G4bool GetAutoRotation() const     	  { return m_flagAutoRotation;}
  //! Get the angle for the first copy
  inline G4double GetFirstAngle() const           { return m_firstAngle;}
  //! Get the total angular span
  inline G4double GetAngularSpan() const          { return m_angularSpan;}
  //! Get the angular pitch between copies
  inline G4double  GetAngularPitch_1() const        { return m_angularSpan/m_repeatNumber;}
  inline G4double  GetAngularPitch_2() const        { return m_angularSpan/(m_repeatNumber-1);}
  //! Get the periodicity of shift or the number of copies in sequence
  inline G4int GetAngularModuloNumber() const  { return m_moduloNumber;}
  //! Get the total Z shift 1 .. 8 in a sequence
  inline G4double GetZShift1() const { return m_zShift1;}
  inline G4double GetZShift2() const { return m_zShift2;}
  inline G4double GetZShift3() const { return m_zShift3;}
  inline G4double GetZShift4() const { return m_zShift4;}
  inline G4double GetZShift5() const { return m_zShift5;}
  inline G4double GetZShift6() const { return m_zShift6;}
  inline G4double GetZShift7() const { return m_zShift7;}
  inline G4double GetZShift8() const { return m_zShift8;}

  //! Get the Z shift pitch between copies
  //  inline G4double GetZShiftPitch() const { return m_zShiftSpan/((m_moduloNumber<2) ? 1; m_moduloNumber-1);} /* take 1 if<2 */

  //! Set the number of repetitions
  inline void SetRepeatNumber(G4int val)
  { m_repeatNumber = val;  }
  //! Set the starting point of the repetition axis
  inline void SetPoint1(const G4Point3D& val)
  { m_point1 = val;  }
  //! Set the end point of the repetition axis
  inline void SetPoint2(const G4Point3D& val)
  { m_point2 = val;  }
  //! Set the value of the auto-rotation flag
  inline void SetAutoRotation(G4bool val)
  { m_flagAutoRotation = val;  }
  //! Set the angle for the first copy
  inline void SetFirstAngle(G4double val)
  { m_firstAngle = val;  }
  //! Set the total angular span
  inline void SetAngularSpan(G4double val)
  { m_angularSpan = val;   }
  //! Set the periodicity of shift or the number of copies in sequence
  inline void SetModuloNumber(G4int val)
  { m_moduloNumber = val;}

  //! Set the total modulo Z shift Span of a sector (accept 8 shift max.)
  inline void SetZShift1(G4double val)  { m_zShift1 = val;}
  inline void SetZShift2(G4double val)  { m_zShift2 = val;}
  inline void SetZShift3(G4double val)  { m_zShift3 = val;}
  inline void SetZShift4(G4double val)  { m_zShift4 = val;}
  inline void SetZShift5(G4double val)  { m_zShift5 = val;}
  inline void SetZShift6(G4double val)  { m_zShift6 = val;}
  inline void SetZShift7(G4double val)  { m_zShift7 = val;}
  inline void SetZShift8(G4double val)  { m_zShift8 = val;}

  //@}

private:
  //! \name repeater parameters
  //@{
  G4int         m_repeatNumber;     	//!< Number of repetitions
  G4Point3D 	  m_point1;     	//!< starting point of the repetition axis
  G4Point3D 	  m_point2;     	//!< end point of the repetition axis
  G4bool  	  m_flagAutoRotation; 	//!< Auto-rotation flag
  G4double  	  m_firstAngle;       	//!< Angle for the first copy
  G4double  	  m_angularSpan;      	//!< Total angular span
  G4int         m_moduloNumber;	//!< Total angular span
  G4double      m_zShift1;	        //!< Shift Span for module 1
  G4double      m_zShift2;	        //!< Shift Span for module 2
  G4double      m_zShift3;	        //!< Shift Span for module 3
  G4double      m_zShift4;	        //!< Shift Span for module 4
  G4double      m_zShift5;	        //!< Shift Span for module 5
  G4double      m_zShift6;	        //!< Shift Span for module 6
  G4double      m_zShift7;	        //!< Shift Span for module 7
  G4double      m_zShift8;	        //!< Shift Span for module 8

  //@}

  //! Messenger
  GateAngularRepeaterMessenger* m_Messenger;

};

#endif
