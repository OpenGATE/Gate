/* @(#)sino_dets.c	1.1 5/10/90 */

#define abs(a) (((a)<0)?-(a):(a))
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

ve_to_dets( v, e, da, db, ndets)
  int v,e,*da,*db,ndets;
{
	int h,i;

	h=ndets/2;
	i=abs(e)%2;
	*da=(ndets+(e-i)/2+v)%ndets;
	*db=(h-(e+i)/2+v)%ndets;
}

dets_to_ve( da, db, v, e, ndets)
  int da,db,*v,*e,ndets;
{
	int h,x,y,a,b,te;

	h=ndets/2;
	x=max(da,db);
	y=min(da,db);
	a=((x+y+h+1)%ndets)/2;
	b=a+h;
	te=abs(x-y-h);
	if ((y<a)||(b<x)) te = -te;
	*e=te;
	*v=a;
}

#ifdef TEST

main(argc, argv)
  int argc;
  char **argv;
{
	int v,e,da,db,ndets;
	int model, boff, aend, bend, bmax, mpair, abucket, bbucket;
	int deta, detb, adet, bdet, vtest, etest;

	ndets=384;	/* 953 */
	while (1)
	{
	printf("dets_to_ve test...\n");
	while (1)
	{
	  printf("Enter d1,d2,ndets: ");
	  da=db=ndets=0;
	  scanf("%d,%d,%d",&da,&db,&ndets);
	  if (ndets==0) break;
	  dets_to_ve( da,db,&v,&e,ndets);
	  printf("(%d,%d)\n",v,e);
	}
	printf("ve_to_dets test...\n");
	while (1)
	{
	  printf("Enter v,e,ndets: ");
	  v,e,ndets=0;
	  scanf("%d,%d,%d",&v,&e,&ndets);
	  if (ndets==0) break;
	  ve_to_dets( v,e,&da,&db,ndets);
	  printf("(%d,%d)\n", da, db);
	}
	printf("consistancy test...which model? ");
	scanf("%d", &model);
	if ((model==911)||(model==931)||(model==951))
	{ boff=5; aend=10; bend=16; bmax=7; }
	else if ((model==933)||(model==953))
	{ boff=3; aend=8; bend=12; bmax=7; }
	else if (model==831)
	{ boff=3; aend=6; bend=10; bmax=5; }
	else 
	{ printf("*** Unknown model...try 931, 933, 831...\n");
	  break;
	}
	ndets=32*bend;
	mpair=1;
	for(abucket=0;abucket<=aend; abucket++)
	  for (bbucket=abucket+boff; bbucket<min(bend,abucket+boff+bmax);
		 bbucket++)
	    {
		printf("mpair %2d = (%2d,%2d)\n", mpair, abucket, bbucket);
		for (deta=0; deta<32; deta++)
		  for (detb=0; detb<32; detb++)
		    {
			adet=32*abucket+deta;
			bdet=32*bbucket+detb;
			dets_to_ve( adet, bdet, &vtest, &etest, ndets);
			ve_to_dets( vtest, etest, &da, &db, ndets);
			if (!((da==adet && db==bdet)||(da==bdet && db==adet)))
			  printf("...(%d,%d)=>(%d,%d)=>(%d,%d)\n",
				adet,bdet,vtest,etest,da,db);
		    }
		mpair++;
	    }
	}
}
#endif
