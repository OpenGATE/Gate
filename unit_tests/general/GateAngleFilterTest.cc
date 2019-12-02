#include <GateAngleFilter.hh>
#include <G4SystemOfUnits.hh>
#include <catch.hpp>
#include <vector>
#include <G4Track.hh>
#include <G4DynamicParticle.hh>
#include <iostream>
#include <G4Proton.hh>


TEST_CASE("checking angle filter","[filter][direction][angular]"){
GateAngleFilter filter("Ane");
G4ThreeVector p(1,1,0);
filter.SetMomentum(p);
filter.SetAngle(50*deg);//degree

INFO("a");



G4ParticleDefinition *proton = G4Proton::ProtonDefinition();
double kEnergy=10*MeV;
G4ThreeVector position(0,0,0);
G4DynamicParticle*  dynPart=new G4DynamicParticle(proton,p,kEnergy);

//G4Track aTrack;
G4Track aTrack(dynPart, 1.0,position);

//Future generate maybe a set of vector in a cone  within the angle to past the test
G4ThreeVector pTrack(0.8,0.2,0);
aTrack.SetMomentumDirection(pTrack);

INFO("test");
double mX= aTrack.GetMomentumDirection().getX();
double mY= aTrack.GetMomentumDirection().getY();
double mZ= aTrack.GetMomentumDirection().getZ();
 

INFO("accepted  momentum "<<mX<<","<<mY<<","<<mZ);
CHECK(filter.Accept(&aTrack));


pTrack.setX(0);
pTrack.setY(0);
pTrack.setZ(1);
aTrack.SetMomentumDirection(pTrack);
mX= aTrack.GetMomentumDirection().getX();
mY= aTrack.GetMomentumDirection().getY();
mZ= aTrack.GetMomentumDirection().getZ();


INFO("rejected  momentum "<<mX<<","<<mY<<","<<mZ);





CHECK( not filter.Accept(&aTrack));


}
// vim: et:sw=2:ai:smartindent
