#ifndef CALFACTOR_HH
#define CALFACTOR_HH

class Calfactor {
public:
  Calfactor(u8 itsNbSct = 6,
	    u8 itsNbMod = 3, u8 itsNbCry = 64, u8 itsNbLay = 2);
  ~Calfactor();
  double operator() (const u8 &, const u8 &, const u8 &, const u8 &) const;
  double &operator()(const u8 &, const u8 &, const u8 &, const u8 &);
  void ReadCalfactorTable(u8 sct, u8 mod, u8 lay, u8 set = 1);
  void ReadAllCalfactorTables(u8 set = 1);
  void WriteAllCalfactorInFiles(u8 set = 1);
private:
  u8 m_nbsct;
  u8 m_nbmod;
  u8 m_nbcry;
  u8 m_nblay;
  double ****m_calfactors;
  mean_std *m_calBase;

  u8 SectorToIndex(const u8 & sct) const;
  u8 IndexToSector(const u8 & idx) const;
};

#endif
