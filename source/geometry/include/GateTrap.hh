/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTrap_h
#define GateTrap_h 1

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

#include "G4Trap.hh"
#include "G4RunManager.hh"
#include "G4VVisManager.hh"

#include "globals.hh"

class GateTrapMessenger;
class G4Material;


class GateTrap  : public GateVVolume
{
  public:
  
    GateTrap(const G4String& itsName,
		 G4bool acceptsChildren=true, 
		 G4int depth=0); 
		 
		 
    GateTrap(const G4String& itsName,const G4String& itsMaterialName,
    			G4double itsDz,G4double itsTheta, G4double itsPhi,
			G4double itsDy1,G4double itsDx1,G4double itsDx2,G4double itsAlp1,
			G4double itsDy2,G4double itsDx3,G4double itsDx4,G4double itsAlp2);
    virtual ~GateTrap();

    FCT_FOR_AUTO_CREATOR_VOLUME(GateTrap)
    
  public:
     virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool flagUpdateOnly);     
     virtual void DestroyOwnSolidAndLogicalVolume();
     virtual void DescribeMyself(size_t indent);
     
     inline G4double GetHalfDimension(size_t axis) 
     { if (axis==2)
      	  return m_TrapDz;
       else
       { if (axis==1)
      	     return m_TrapDy1;
	 else
	     return m_TrapDx1;
       }
     }

  public:
     inline G4double GetTrapDz()          	      {return m_TrapDz;}
     inline G4double GetTrapTheta()          	      {return m_TrapTheta;}
     inline G4double GetTrapPhi()          	      {return m_TrapPhi;}
     inline G4double GetTrapDy1()          	      {return m_TrapDy1;}
     inline G4double GetTrapDx1()          	      {return m_TrapDx1;}
     inline G4double GetTrapDx2()          	      {return m_TrapDx2;}
     inline G4double GetTrapAlp1()          	      {return m_TrapAlp1;}
     inline G4double GetTrapDy2()          	      {return m_TrapDy2;}
     inline G4double GetTrapDx3()          	      {return m_TrapDx3;}
     inline G4double GetTrapDx4()          	      {return m_TrapDx4;}
     inline G4double GetTrapAlp2()          	      {return m_TrapAlp2;}

     inline void SetTrapDz(G4double val)      	      {m_TrapDz  = val; /*ComputeParameters();*/}
     inline void SetTrapTheta(G4double val)	      {m_TrapTheta  = val; /*ComputeParameters();*/}
     inline void SetTrapPhi(G4double val)      	      {m_TrapPhi  = val; /*ComputeParameters();*/}
     inline void SetTrapDy1(G4double val)      	      {m_TrapDy1 = val; /*ComputeParameters();*/}
     inline void SetTrapDx1(G4double val)      	      {m_TrapDx1 = val; /*ComputeParameters();*/}
     inline void SetTrapDx2(G4double val)      	      {m_TrapDx2 = val; /*ComputeParameters();*/}
     inline void SetTrapAlp1(G4double val)            {m_TrapAlp1 = val; /*ComputeParameters();*/}
     inline void SetTrapDy2(G4double val)      	      {m_TrapDy2 = val; /*ComputeParameters();*/}
     inline void SetTrapDx3(G4double val)      	      {m_TrapDx3 = val; /*ComputeParameters();*/}
     inline void SetTrapDx4(G4double val)      	      {m_TrapDx4 = val; /*ComputeParameters();*/}
     inline void SetTrapAlp2(G4double val)            {m_TrapAlp2 = val; /*ComputeParameters();*/}


  private:
     G4double m_TrapDz,m_TrapTheta,m_TrapPhi,m_TrapDy1,m_TrapDx1,m_TrapDx2,m_TrapAlp1;
     G4double m_TrapDy2,m_TrapDx3,m_TrapDx4,m_TrapAlp2;

     G4Trap*              m_Trap_solid;
     G4LogicalVolume*     m_Trap_log;

     GateTrapMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(trap,GateTrap)
#endif

