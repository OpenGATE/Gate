/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateMessenger_h
#define GateMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

/*! \class GateMessenger
    \brief Provides some basic fonctionalities for GATE messengers

    - GateMessenger - by Daniel.Strul@iphe.unil.ch

    - The main function of a GateMessenger is to create (as needed) and manage a UI messenger
      directory.

    - It stores the directory base-name and directory-name ( = "/gate/" + base-name), and
      a pointer to the directory

    - It also provides the ability to communicate with the geometry, and to ask it to update
      or rebuild itself (OK... storing this method here is not very OO...)
*/
class GateMessenger: public G4UImessenger
{
  public:
    //! Constructor
    //! 'itsName' is the base-name of the directory
    GateMessenger(const G4String& itsName,G4bool createDirectory=true);
    virtual ~GateMessenger();

    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the directory base-name
    inline const G4String& GetName() const
      { return mName; }
    //! Returns the directory full-path ( = "/Gate/" + base-name)
    inline const G4String& GetDirectoryName() const
      { return mDirName; }
    //! Returns the directory
    inline G4UIdirectory* GetDirectory() const
      { return pDir; }
    //! Adds a new line to the directory guidance
    void SetDirectoryGuidance(const G4String& guidance);

  protected:
    //! Tells GATE that the geometrical parameters have undergone a minor modification
    //! so that the geometry should be updated
//    virtual void TellGeometryToUpdate();

    //! Tells GATE that the geometrical parameters have undergone a major modification
    //! so that the geometry should be rebuilt
//    virtual void TellGeometryToRebuild();

    //! Tool method: compute a directory full-path based on the geometry base-name
    inline static G4String ComputeDirectoryName(const G4String& name)
      { return G4String("/gate/") + name + "/" ; }

  protected:
    G4String          	      mName; 	    //|< Directory base-name

    G4String  	              mDirName;    //!< Directory full-path ( = "/gate/" + base-name)

    G4UIdirectory*            pDir;  	    //! Ptr to the directory

};

#endif
