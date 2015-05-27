/**
 * \class       GateImageBox
 * \brief       Displays an image with OpenGL.
 * \details     Behaves like a G4Box for tracking but displays image with three orthogonal planes in OpenGL: axial, sagittal and coronal.
 * \author      Jérôme Suhard <jerome@suhard.fr>
 * \date        04/2014
 * \warning     Only works in Immediate mode ( /vis/open OGLI ).
 * \copyright   GNU Lesser General Public Licence (LGPL).
 */

#ifndef GATEIMAGEBOX_HH
#define GATEIMAGEBOX_HH

#ifdef G4VIS_USE_OPENGL

  #include "G4Version.hh"
  /**
   * Only Geant4 >= 9.6.0 had the G4OpenGLSceneHandler::GetObjectTransformation() method
   */
  #if G4VERSION_NUMBER >= 960

    #include "GateConfiguration.h"

    #ifdef GATE_USE_OPENGL
      #ifdef __APPLE__
        #include <OpenGL/gl.h>
      #else
        #include <GL/gl.h>
      #endif

      #define G4VIS_BUILD_OPENGL_DRIVER
      #include "G4OpenGLSceneHandler.hh"

      #define GATEIMAGEBOX_USE_OPENGL 1
    #endif /* GATE_USE_OPENGL */

  #endif /* G4VERSION_NUMBER */

#endif /* G4VIS_USE_OPENGL */

#include "G4Box.hh"

#include "GateImage.hh"

class GateImageBox : public virtual G4Box {

public:
    GateImageBox(const GateImage & image, const G4String & name);
    ~GateImageBox();

  typedef float PixelType; // FIXME
  typedef std::vector<PixelType> PixelTypeVec;

    void DescribeYourselfTo(G4VGraphicsScene& scene) const;
private:
#ifdef GATEIMAGEBOX_USE_OPENGL
    void DescribeYourselfTo(G4OpenGLSceneHandler& scene) const;

    PixelTypeVec getXYSlice(const GateImage & image, const size_t z) const;
    PixelTypeVec getXZSlice(const GateImage & image, const size_t y) const;
    PixelTypeVec getYZSlice(const GateImage & image, const size_t x) const;
    GLubyte * convertToRGB(PixelTypeVec slice, PixelType min, PixelType max) const;
    GLuint genOpenGLTexture(const GLubyte * rgb, int width, int height) const;
    void initOpenGLTextures(const GateImage & image, const size_t x, const size_t y, const size_t z);

    size_t position_x;
    size_t position_y;
    size_t position_z;
    size_t resolution_x;
    size_t resolution_y;
    size_t resolution_z;
    GLuint texture_xy;
    GLuint texture_xz;
    GLuint texture_yz;
#endif
};

#endif /* GATEIMAGEBOX_HH */
