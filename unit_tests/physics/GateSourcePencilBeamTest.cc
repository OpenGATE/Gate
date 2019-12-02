#include <catch.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <GateSourcePencilBeam.hh>
#include <G4SystemOfUnits.hh>
#include <G4Event.hh>
#include <GateRunManager.hh>
#include <GatePhysicsList.hh>

#include <Stat.h>

namespace unit_tests {

TEST_CASE("pencil beam test", "[example][physics]") {

INFO("create geant4 environment for particle sources");
GateRunManager runManager;
runManager.SetUserInitialization(GatePhysicsList::GetInstance());

INFO("create pencil beam source");
GateSourcePencilBeam gpb("test", false);

INFO("set kinematics");
gpb.SetIonParameter("proton");
gpb.SetPosition(G4ThreeVector(10. * mm, 20. * mm, 5. * mm));
gpb.SetSigmaX(1. * mm);
gpb.SetSigmaY(1. * mm);
gpb.SetSigmaTheta(1. * mrad);
gpb.SetSigmaPhi(1. * mrad);
gpb.SetEllipseXThetaArea(1. * mm * mrad);
gpb.SetEllipseYPhiArea(1. * mm * mrad);
gpb.SetEnergy(100 * MeV);
gpb.SetSigmaEnergy(0.1 * MeV);

varstat1D Estat("E");
varstat2D XYstat("X", "Y");
varstat2D XDXstat("X", "DX");
varstat2D YDYstat("Y", "DY");
for (int i = 0; i < 1000; ++i) {
  INFO("create event");
  G4Event event;

  INFO("generate primaries");
  G4int nret = gpb.GeneratePrimaries(&event);

  INFO("number of generated primaries should be 1");
  REQUIRE(nret == 1);

  INFO("number of vertices in event should also be 1");
  G4int nvertex = event.GetNumberOfPrimaryVertex();
  REQUIRE(nvertex == 1);
  const G4PrimaryVertex *v = event.GetPrimaryVertex();
  double x = v->GetX0();
  double y = v->GetY0();
  double z = v->GetZ0();

  INFO("number of particles in vertex should also be 1");
  G4int nprimaries = v->GetNumberOfParticle();
  REQUIRE(nprimaries == 1);
  CHECK(z / mm == Approx(5.).margin(1e-10));

  const G4PrimaryParticle *p = v->GetPrimary();
  double dx = p->GetMomentumDirection()[0];
  double dy = p->GetMomentumDirection()[1];
  double dz = p->GetMomentumDirection()[2];
  CHECK(dx * dx + dy * dy + dz * dz == Approx(1.).margin(1.e-6));
  Estat.add_value(p->GetKineticEnergy());
  XYstat.add_value(x, y);
  XDXstat.add_value(x, atan(dx));
  YDYstat.add_value(y, atan(dy));
}
INFO("energy spread should be 0.1 MeV");
CHECK(Estat.rms() / MeV == Approx(0.1).margin(0.01));
INFO("average energy should be 100 MeV within 0.1 sigma");
CHECK(Estat.average() / MeV == Approx(100.).margin(0.01));
INFO("rms X and Y should be 1 mm");
CHECK(XYstat.rmsX() / mm == Approx(1.).margin(0.1));
CHECK(XYstat.rmsY() / mm == Approx(1.).margin(0.1));
INFO("average X and should be 10 mm and 20 mm within 0.1 sigma");
CHECK(XYstat.averageX() / mm == Approx(10.).margin(0.1));
CHECK(XYstat.averageY() / mm == Approx(20.).margin(0.1));
INFO("rms theta and phi should be 1 mrad");
CHECK(XDXstat.rmsY() / mrad == Approx(1.).margin(0.1));
CHECK(YDYstat.rmsY() / mrad == Approx(1.).margin(0.1));
INFO("average theta and phi should be 0 mrad");
CHECK(XDXstat.averageY() / mrad == Approx(0.).margin(0.1));
CHECK(YDYstat.averageY() / mrad == Approx(0.).margin(0.1));
INFO("emittance in X (rho=" << XDXstat.rho() << ") and Y (rho=" << YDYstat.rho() << ")");
REQUIRE(pow(XDXstat.rho(), 2.) < 1);
REQUIRE(pow(YDYstat.rho(), 2.) < 1);
CHECK(pi * sqrt(1. - pow(XDXstat.rho(), 2.)) == Approx(1.).margin(0.1));
CHECK(pi * sqrt(1. - pow(YDYstat.rho(), 2.)) == Approx(1.).margin(0.1));
}

}