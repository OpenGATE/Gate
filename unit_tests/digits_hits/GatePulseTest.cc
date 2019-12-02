#include <catch.hpp>
#include <GatePulse.hh>

#include <Stat.h>

namespace unit_tests {

TEST_CASE("GatePulse centroid merge","[example][digits_hits]"){
    GatePulse pulse1;
    pulse1.SetEnergy(10.0);
    pulse1.SetTime(100.0);
    pulse1.SetLocalPos(G4ThreeVector(1, 2, 3));
    pulse1.SetGlobalPos(G4ThreeVector(10, 20, 30));
    pulse1.SetNPhantomCompton(3);
    pulse1.SetNPhantomRayleigh(7);
    pulse1.SetNSeptal(3);

    GatePulse pulse2;
    pulse2.SetEnergy(15.0);
    pulse2.SetTime(150.0);
    pulse2.SetLocalPos(G4ThreeVector(2, 3, 4));
    pulse2.SetGlobalPos(G4ThreeVector(20, 30, 40));
    pulse2.SetNPhantomCompton(4);
    pulse2.SetNPhantomRayleigh(6);
    pulse2.SetNSeptal(5);

    pulse1.CentroidMerge(&pulse2);

    REQUIRE(pulse1.GetTime() == Approx(100.0).epsilon(0.0001));
    REQUIRE(pulse1.GetEnergy() == Approx(25.0).epsilon(0.0001));
    REQUIRE(pulse1.GetLocalPos() == G4ThreeVector(1.6, 2.6, 3.6));
    REQUIRE(pulse1.GetGlobalPos() == G4ThreeVector(16.0, 26.0, 36.0));
    REQUIRE(pulse1.GetNPhantomCompton() == 4);
    REQUIRE(pulse1.GetNPhantomRayleigh() == 7);
    REQUIRE(pulse1.GetNSeptal() == 5);
}

TEST_CASE("GatePulse centroid merge compton","[example][digits_hits]"){
    GatePulse pulse1;
    pulse1.SetEnergy(10.0);
    pulse1.SetTime(100.0);
    pulse1.SetLocalPos(G4ThreeVector(1, 2, 3));
    pulse1.SetGlobalPos(G4ThreeVector(10, 20, 30));
    pulse1.SetNPhantomCompton(3);
    pulse1.SetNPhantomRayleigh(7);
    pulse1.SetNSeptal(3);

    GatePulse pulse2;
    pulse2.SetEnergy(15.0);
    pulse2.SetTime(150.0);
    pulse2.SetLocalPos(G4ThreeVector(2, 3, 4));
    pulse2.SetGlobalPos(G4ThreeVector(20, 30, 40));
    pulse2.SetNPhantomCompton(4);
    pulse2.SetNPhantomRayleigh(6);
    pulse2.SetNSeptal(5);

    pulse1.CentroidMergeCompton(&pulse2);

    REQUIRE(pulse1.GetTime() == Approx(100.0).epsilon(0.0001));
    REQUIRE(pulse1.GetEnergy() == Approx(25.0).epsilon(0.0001));
    REQUIRE(pulse1.GetLocalPos() == G4ThreeVector(1, 2, 3));
    REQUIRE(pulse1.GetGlobalPos() == G4ThreeVector(10, 20, 30));
    REQUIRE(pulse1.GetNPhantomCompton() == 4);
    REQUIRE(pulse1.GetNPhantomRayleigh() == 7);
    REQUIRE(pulse1.GetNSeptal() == 5);
}

}