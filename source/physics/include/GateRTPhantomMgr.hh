#ifndef GateRTPhantomMgr_H
#define GateRTPhantomMgr_H

#include <vector>
#include "GateRTPhantom.hh"

class GateRTPhantomMgrMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateRTPhantomMgr
{
public:

  virtual ~GateRTPhantomMgr(); //!< Destructor

  void AddPhantom( G4String aname );
  void UpdatePhantoms(G4int cK);
  void UpdatePhantoms(G4double aTime);

  GateRTPhantom * CheckSourceAttached( G4String aname);

  GateRTPhantom * CheckGeometryAttached( G4String aname);


  //! Used to create and access the OutputMgr
  static GateRTPhantomMgr* GetInstance() {
    if (instance == 0)
      instance = new GateRTPhantomMgr("RTPhantom");
    return instance;
  };

  void SetVerboseLevel(G4int val);
  inline G4int GetVerboseLevel(){return m_verboseLevel;};

  //! Provides a description of the properties of the Mgr and of its output modules 
  void Describe();

  //! Getter used by the Messenger to construct the commands directory
  inline G4String GetName()              { return m_name; };
  inline void     SetName(G4String name) { m_name = name; };
  GateRTPhantom* Find( G4String aname);
private:

  GateRTPhantomMgr(const G4String name); 
  static GateRTPhantomMgr* instance;

  //! Verbose level
  G4int                      m_verboseLevel;

  //! List of the output modules
  std::vector<GateRTPhantom*>   m_RTPhantom;

  //! messenger for the Mgr specific commands
  GateRTPhantomMgrMessenger*    m_messenger;

  //! class name, used by the messenger
  G4String                   m_name;

};

#endif
