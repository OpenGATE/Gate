#include <catch.hpp>
#include <GateSPSPosDistribution.hh>
#include <G4SPSRandomGenerator.hh>

#include <Stat.h>

namespace unit_tests {

TEST_CASE("Point position distributions","[example][physics]"){

   G4SPSRandomGenerator rnd_gen;
   G4ThreeVector point(1.1, 10.0, 33.0);

   GateSPSPosDistribution p_dist;
   p_dist.SetBiasRndm( &rnd_gen );
   p_dist.SetPosDisType(G4String("Point"));
   p_dist.SetCentreCoords(point);

   SECTION("A randomly generated point from a point distribution is the point itself") {
       REQUIRE(p_dist.GenerateOne() == point);
       REQUIRE_FALSE(p_dist.GenerateOne() == (point + G4ThreeVector(1.0, 0, 0)));
   }

   SECTION("Rotating a point should not change its position") {
       p_dist.SetPosRot1(G4ThreeVector(0, 1, 0));
       p_dist.SetPosRot2(G4ThreeVector(1, 1, 0));
       REQUIRE(p_dist.GenerateOne() == point);
   }
}

TEST_CASE("Plane position distributions","[example][physics]"){

    G4SPSRandomGenerator rnd_gen;
    G4ThreeVector centre_point(3.0, 10.0, 30.0);

    GateSPSPosDistribution p_dist;
    p_dist.SetBiasRndm( &rnd_gen );
    p_dist.SetPosDisType(G4String("Plane"));
    p_dist.SetPosDisShape(G4String("Circle"));
    p_dist.SetCentreCoords(centre_point);
    p_dist.SetRadius(2.0);

    for (int i = 0; i < 100; ++i) {
       G4ThreeVector p_rnd = (p_dist.GenerateOne() - centre_point);
       REQUIRE(p_rnd.getZ() == Approx(0.0).margin(0.0001));
       REQUIRE(p_rnd.getR() <= 2.0);
    }

    // todo: add rotation tests
}

TEST_CASE("Volume position distributions","[example][physics]"){

    G4SPSRandomGenerator rnd_gen;
    G4ThreeVector centre_point(3.0, 10.0, 30.0);

    GateSPSPosDistribution p_dist;
    p_dist.SetBiasRndm( &rnd_gen );
    p_dist.SetPosDisType(G4String("Volume"));
    p_dist.SetPosDisShape(G4String("Sphere"));
    p_dist.SetCentreCoords(centre_point);
    p_dist.SetRadius(2.0);

    for (int i = 0; i < 100; ++i) {
        G4ThreeVector p_rnd = (p_dist.GenerateOne() - centre_point);
        REQUIRE(p_rnd.getR() <= 2.0);
    }
}

TEST_CASE("Gaussian beam","[example][physics]") {
        G4SPSRandomGenerator rnd_gen;
        G4ThreeVector centre_point(3.0, 10.0, 30.0);

        GateSPSPosDistribution p_dist;
        p_dist.SetBiasRndm( &rnd_gen );
        p_dist.SetPosDisType(G4String("Beam"));
        p_dist.SetCentreCoords(centre_point);

        p_dist.SetBeamSigmaInX(2.0);
        p_dist.SetBeamSigmaInY(5.0);

        varstat1D x_dist("X");
        varstat1D y_dist("Y");
        varstat1D z_dist("Z");
        for (int i = 0; i < 1000; ++i) {
            G4ThreeVector p_rnd = (p_dist.GenerateOne() - centre_point);
            x_dist.add_value(p_rnd.getX());
            y_dist.add_value(p_rnd.getY());
            z_dist.add_value(p_rnd.getZ());
        }
        REQUIRE(x_dist.average() == Approx(0.0).margin(0.2)); // these margins are pretty large
        REQUIRE(y_dist.average() == Approx(0.0).margin(0.2));
        REQUIRE(z_dist.average() == Approx(0.0).margin(0.2));
        REQUIRE(x_dist.rms() == Approx(2.0).margin(0.2));
        REQUIRE(y_dist.rms() == Approx(5.0).margin(0.2));
        REQUIRE(z_dist.rms() == Approx(0.0).margin(0.2));
}

// todo: add tests for positrontype

}