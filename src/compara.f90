!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!*********************************************************************!
!*                                                                  **!
!*                           compara                                **!
!*                      ====================                        **!
!*                                                                  **!
!*  Ricardo Mendes Ribeiro                                          **!
!*  Date: Jan, 2020                                                 **!
!*  Description: Main program that compares the wavefunctions       **!
!*             of a set                                             **!
!*                                                                  **!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!
!************************************************************************

MODULE COMUM

  IMPLICIT NONE

  INTEGER :: numero_kx, numero_ky, numero_kz
  INTEGER :: nbands, nks, nr
  INTEGER :: startpoint, sym1, sym2
  INTEGER(KIND=4),ALLOCATABLE :: n0(:),n1(:),n2(:),n3(:)
  INTEGER(KIND=4),ALLOCATABLE :: connections(:,:,:), connections2(:,:,:)
  INTEGER(KIND=4),ALLOCATABLE :: planes(:,:), quality(:,:)
  REAL(KIND=8),ALLOCATABLE :: eigenvalues(:,:)
  REAL :: tol

  CHARACTER(LEN=20),PARAMETER :: &
  fmt1 = '(3f14.8,3f22.16)', &
  fmt2 = '(i4)', &
  fmt3 = '(4i6)', &
  fmt4 = '(5i6)', &
  fmt5 = '(6i6)'

  LOGICAL,PARAMETER :: usesymmetry = .TRUE.
  CHARACTER(LEN=50),PARAMETER :: wfcdirectory = 'wfc'

END MODULE COMUM
! ****************************************************************************

SUBROUTINE atribute(sym)
! ****************************************************************************

  USE COMUM

  INTEGER(KIND=4) :: i,j, nk, nk1, nk0, banda
  INTEGER :: sym

  DO banda = 1,nbands
    IF (planes(sym,banda) == 0) THEN
      planes(sym,banda) = planes(startpoint,banda)      ! Starts at k-point sym
    ELSEIF (planes(sym,banda) .NE. planes(startpoint,banda)) THEN
      WRITE(*,*) 'ALERT: serious inconsistency',sym,banda,startpoint,banda
    ENDIF
    nk1 = sym
    DO WHILE (n3(nk1-numero_kx) .NE. -1)                ! Runs until finds end of +y
      nk = nk1
      DO WHILE (n2(nk-1) .NE. -1)                       ! Runs until finds end of +x
        CALL att(nk,banda)
        nk = nk + 1
      ENDDO
      nk = nk1 - 1
      DO WHILE (n0(nk+1) .NE. -1)                       ! Runs until finds end of -x
        CALL att(nk,banda)
        nk = nk - 1
      ENDDO
      nk1 = nk1 + numero_kx
    ENDDO

    nk1 = startpoint - numero_kx
    DO WHILE (n1(nk1+numero_kx) .NE. -1)                ! Runs until finds end of -y
      nk = nk1
      DO WHILE (n2(nk-1) .NE. -1)                       ! Runs until finds end of +x
        CALL att(nk,banda)
        nk = nk + 1
      ENDDO
      nk = nk1 - 1
      DO WHILE (n0(nk+1) .NE. -1)                       ! Runs until finds end of -x
        CALL att(nk,banda)
        nk = nk - 1
      ENDDO
      nk1 = nk1 - numero_kx
    ENDDO

    nk0 = -1                                            ! Draws filled planes
    WRITE(*,*)
    WRITE(*,*)banda, '         | y  x ->'
    DO j = 0,numero_ky-1
      WRITE(*,*)
      DO i = 0,numero_kx-1
        nk0 = nk0 + 1
        WRITE(*,fmt2,advance="no") planes(nk0,banda)    ! prints attributed bands to output
      ENDDO
      WRITE(*,*)
    ENDDO
  ENDDO


END SUBROUTINE atribute
! ****************************************************************************

SUBROUTINE att(nk,banda)
! ****************************************************************************

  USE COMUM
  INTEGER(KIND=4) :: nk, banda, banda0

  banda0 = planes(nk,banda)                      ! This is the machine nr of nk that belongs to banda
  IF (banda0 == 0) RETURN                        ! if it is not defined to which it belongs, returns
  IF (connections(nk,banda0,0) > 0) THEN         ! if there is a connection to n0(nk),banda0
    IF (planes(n0(nk), banda) == 0) THEN         ! and n0(nk),banda is free
      planes(n0(nk), banda) = connections(nk,banda0,0)     ! attribute machine nr to n0(nk),banda
      quality(n0(nk), banda) = 1
    ENDIF
  ENDIF
  IF (connections(nk,banda0,1) > 0) THEN
    IF (planes(n1(nk), banda) == 0) THEN
      planes(n1(nk), banda) = connections(nk,banda0,1)
      quality(n1(nk), banda) = 1
    ENDIF
  ENDIF
  IF (connections(nk,banda0,2) > 0) THEN
    IF (planes(n2(nk), banda) == 0) THEN
      planes(n2(nk), banda) = connections(nk,banda0,2)
      quality(n2(nk), banda) = 1
    ENDIF
  ENDIF
  IF (connections(nk,banda0,3) > 0) THEN
    IF (planes(n3(nk), banda) == 0) THEN
      planes(n3(nk), banda) = connections(nk,banda0,3)
      quality(n3(nk), banda) = 1
    ENDIF
  ENDIF

END SUBROUTINE att
! ****************************************************************************

SUBROUTINE eig_compare
! ****************************************************************************

  USE COMUM

  IMPLICIT NONE

  INTEGER(KIND=4) :: nk, banda, banda1
  INTEGER :: ii, jj, ll, hh, i
  INTEGER :: viz(0:3), val(0:3)
  REAL(KIND=8) :: eig(0:3), e0, e1


  DO nk = 0,nks-1
    IF (COUNT((planes(nk,:)==0))==0) CYCLE
    DO banda = 1,nbands
      IF (planes(nk,banda) == 0) THEN    ! finds empty space: has to compare with neighbors
        viz = 0                          ! saves machine nr of surrounding k points of banda
        eig = -100.0
        val = 0
        IF (n0(nk) > 0) THEN
          viz(0) = planes(n0(nk),banda)
          IF (viz(0) .NE. 0) eig(0) = eigenvalues(n0(nk),viz(0))
        ENDIF
        IF (n1(nk) > 0) THEN
          viz(1) = planes(n1(nk),banda)
          IF (viz(1) .NE. 0) eig(1) = eigenvalues(n1(nk),viz(1))
        ENDIF
        IF (n2(nk) > 0) THEN
          viz(2) = planes(n2(nk),banda)
          IF (viz(2) .NE. 0) eig(2) = eigenvalues(n2(nk),viz(2))
        ENDIF
        IF (n3(nk) > 0) THEN
          viz(3) = planes(n3(nk),banda)
          IF (viz(3) .NE. 0) eig(3) = eigenvalues(n3(nk),viz(3))
        ENDIF
        DO banda1 = nbands,1,-1
          IF (COUNT((planes(nk,:)==banda1)) == 0) THEN
            e0 = eigenvalues(nk,banda1) - tol
            e1 = eigenvalues(nk,banda1) + tol
            IF (eig(0) > e0 .AND. eig(0) < e1) val(0) = banda1
            IF (eig(1) > e0 .AND. eig(1) < e1) val(1) = banda1
            IF (eig(2) > e0 .AND. eig(2) < e1) val(2) = banda1
            IF (eig(3) > e0 .AND. eig(3) < e1) val(3) = banda1
          ENDIF
        ENDDO

        ii = 0
        jj = 0
        ll = 0
        hh = 0
        SELECT CASE (COUNT((val(:) == 0)))
        CASE (0)
          DO i = 0,3
            IF (val(i) == 0) CYCLE
            IF (ii == 0) THEN
              ii = val(i)
            ELSEIF (jj == 0) THEN
              jj = val(i)
            ELSEIF (ll == 0) THEN
              ll = val(i)
            ELSE
              hh = val(i)
            ENDIF
          ENDDO
          IF (ii == jj .OR. ii == ll .OR. ii == hh) THEN
            planes(nk,banda) = ii
            quality(nk,banda) = 11
          ELSEIF (jj == ll .OR. jj == hh) THEN
            planes(nk,banda) = jj
            quality(nk,banda) = 11
          ELSEIF (ll == hh) THEN
            planes(nk,banda) = ll
            quality(nk,banda) = 11
          ELSE
            planes(nk,banda) = ii
            quality(nk,banda) = 12
          ENDIF

        CASE (1)
          DO i = 0,3
            IF (val(i) == 0) CYCLE
            IF (ii == 0) THEN
              ii = val(i)
            ELSEIF ( jj== 0 ) THEN
              jj = val(i)
            ELSE
              ll = val(i)
            ENDIF
          ENDDO
          IF (ii == jj .OR. ii == ll) THEN
            planes(nk,banda) = ii
            quality(nk,banda) = 11
          ELSEIF (jj == ll) THEN
            planes(nk,banda) = jj
            quality(nk,banda) = 11
          ELSE
            planes(nk,banda) = ii
            quality(nk,banda) = 12
          ENDIF

       CASE (2)
          DO i = 0,3
            IF (val(i) == 0) CYCLE
            IF (ii == 0) THEN
              ii = val(i)
            ELSE
              jj = val(i)
            ENDIF
          ENDDO
          IF (ii == jj) THEN
            planes(nk,banda) = ii
            quality(nk,banda) = 11
          ELSE
            planes(nk,banda) = ii
            quality(nk,banda) = 12
          ENDIF

        CASE (3)
          planes(nk,banda) = MAXVAL(val)
          quality(nk,banda) = 11

        CASE (4)
          WRITE(*,*) 'No nearby band!'
        END SELECT

      ENDIF
    ENDDO
  ENDDO

END SUBROUTINE eig_compare



! ****************************************************************************
! ****************************************************************************
PROGRAM compara
! ****************************************************************************
! ****************************************************************************

  USE COMUM

  IMPLICIT NONE

  INTEGER(KIND=4) :: i,j, nk, nk1, banda, banda0
  INTEGER ::  numberof0, delta, iter, iter1
  INTEGER :: lin, a
  REAL :: col
  INTEGER :: npoints, points(1:10)    ! Number of seeding points and their number
  REAL(KIND=8) :: e0, e1

  lin = 2
  col = 0.1
  tol = 0.05                          ! energy tolerance to admit a point to a band

  WRITE(*,*) ' Reading from file connections.dat'
  OPEN(FILE='connections.dat', UNIT=2,STATUS="OLD")
  READ(2,*) numero_kx, numero_ky, numero_kz
  WRITE(*,*) numero_kx, numero_ky, numero_kz
  READ(2,*) nbands
  READ(2,*) nks
  READ(2,*) nr
  WRITE(*,*) ' Number of bands ',nbands
  WRITE(*,*) ' Number of k-points: ', nks
  WRITE(*,*) ' Size of wfc: ', nr
  WRITE(*,*)
  CLOSE(UNIT=2)

  IF (usesymmetry) THEN
    WRITE(*,*) ' Using symmetry'

    OPEN(FILE='points.dat', UNIT=2,STATUS="OLD")
    READ(2,*) npoints
    WRITE(*,*) npoints, ' startimg seeds'
    DO i = 1,npoints
      READ(2,*) points(i)
      WRITE(*,*) points(i)
    ENDDO
    CLOSE(UNIT=2)
    startpoint = points(1)
  ELSE
    startpoint = NINT(numero_kx*(lin+col))
    WRITE(*,*) ' Not using symmetry'
  ENDIF

  ALLOCATE(n0(0:nks),n1(0:nks),n2(0:nks),n3(0:nks))
  ALLOCATE(connections(0:nks-1,1:nbands,0:3))
  ALLOCATE(connections2(0:nks-1,1:nbands,0:3))
  ALLOCATE(planes(0:nks-1,1:nbands),quality(0:nks-1,1:nbands))
  ALLOCATE(eigenvalues(0:nks-1,1:nbands))

  OPEN(UNIT=2,FILE='wfc/eigenvalues',STATUS='OLD')
  DO i = 0,nks-1
    READ(2,*) a, eigenvalues(i,:)
  ENDDO

  planes = 0
  quality = 0
  connections = 0

! ****************************************************************************
  WRITE(*,*)' Start reading files'
  OPEN(UNIT=3,FILE='neighbors',STATUS='OLD')
  DO i = 0,nks-1
    READ(3,fmt4) nk,n0(i),n1(i),n2(i),n3(i)
  ENDDO
  CLOSE(UNIT=3)

  OPEN(UNIT=9,FILE='connections',STATUS='OLD')
  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      READ(9,fmt5)nk,banda0,connections(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)

  OPEN(UNIT=9,FILE='connections2',STATUS='OLD')
  DO nk1 = 0,nks-1
    DO banda = 1,nbands
      READ(9,fmt5)nk,banda0,connections2(nk1,banda,:)
    ENDDO
  ENDDO    
  CLOSE(UNIT=9)

  WRITE(*,*)' Finished reading files'
! ****************************************************************************

  DO banda = 1,nbands
    planes(startpoint,banda) = banda
    quality(startpoint,banda) = 1
  ENDDO

  CALL atribute(startpoint)

  IF (usesymmetry) THEN
   DO i = 2,npoints
    CALL atribute(points(i))
   ENDDO
  ENDIF


! ****************************************************************************
  numberof0 = COUNT((planes(:,:)==0))
  delta = 1
  iter = 0
DO WHILE (delta .NE. 0)
  iter = iter + 1
  WRITE(*,*)
  WRITE(*,*) ' Number of zeros ', numberof0
  WRITE(*,*) '*** ITERATION ****************** ',iter
  nk = -1
  DO j = 0,numero_ky-1
    DO i = 0,numero_kx-1
      nk = nk + 1
      IF (COUNT((planes(nk,:)==0))==0) CYCLE
      WRITE(*,*)
      WRITE(*,*)' *********************** K-point nr. ',nk,' *****' !,' coord. ',i,j
      DO banda = 1,nbands
        IF (planes(nk,banda) == 0) THEN
!          WRITE(*,*) ' Free place at plane ',banda,n0(nk),n1(nk),n2(nk),n3(nk)
!          WRITE(*,*) ' Connections to nk,banda         ',connections(nk,banda,:)

          IF (n0(nk) > 0) THEN
            IF (planes(n0(nk),banda) > 0) THEN
 !           WRITE(*,*)'n0',planes(n0(nk),banda),connections(nk,banda,0)
            IF (connections(n0(nk),planes(n0(nk),banda),2) > 0) THEN
              IF (ANY(planes(nk,:) .NE. connections(n0(nk),planes(n0(nk),banda),2))) THEN  ! Avoid repetitions
                planes(nk,banda) = connections(n0(nk),planes(n0(nk),banda),2)
                quality(nk,banda) = 2
                WRITE(*,*) ' Value in 0- ',n0(nk),banda,' Choose: ', planes(nk,banda)
                CYCLE
              ENDIF
            ENDIF
            ENDIF
          ENDIF

          IF (n1(nk) > 0) THEN
            IF (planes(n1(nk),banda) > 0) THEN
!            WRITE(*,*)'n1',planes(n1(nk),banda),connections(nk,banda,1)
            IF (connections(n1(nk),planes(n1(nk),banda),3) > 0) THEN
              IF (ANY(planes(nk,:) .NE. connections(n1(nk),planes(n1(nk),banda),3))) THEN  ! Avoid repetitions
                planes(nk,banda) = connections(n1(nk),planes(n1(nk),banda),3)
                quality(nk,banda) = 2
                WRITE(*,*) ' Value in 1- ',n1(nk),banda,' choose: ', planes(nk,banda)
                CYCLE
              ENDIF
            ENDIF
            ENDIF
          ENDIF

          IF (n2(nk) > 0) THEN
            IF (planes(n2(nk),banda) > 0) THEN
!            WRITE(*,*)'n2',planes(n2(nk),banda),connections(nk,banda,2)
            IF (connections(n2(nk),planes(n2(nk),banda),0) > 0) THEN
              IF (ANY(planes(nk,:) .NE. connections(n2(nk),planes(n2(nk),banda),0))) THEN  ! Avoid repetitions
                planes(nk,banda) = connections(n2(nk),planes(n2(nk),banda),0)
                quality(nk,banda) = 2
                WRITE(*,*) ' Value in 2- ',n2(nk),banda,' choose: ', planes(nk,banda)
                CYCLE
              ENDIF
            ENDIF
            ENDIF
          ENDIF

          IF (n3(nk) > 0) THEN
            IF (planes(n3(nk),banda) > 0) THEN
!            WRITE(*,*)'n3',planes(n3(nk),banda),connections(nk,banda,3)
            IF (connections(n3(nk),planes(n3(nk),banda),1) > 0) THEN
              IF (ANY(planes(nk,:) .NE. connections(n3(nk),planes(n3(nk),banda),1))) THEN  ! Avoid repetitions
                planes(nk,banda) = connections(n3(nk),planes(n3(nk),banda),1)
                quality(nk,banda) = 2
                WRITE(*,*) ' Value in 3- ',n3(nk),banda,' choose: ', planes(nk,banda)
                CYCLE
              ENDIF
            ENDIF
            ENDIF
          ENDIF

        ENDIF
      ENDDO
    ENDDO
  ENDDO
  delta = numberof0 - COUNT((planes(:,:)==0))
  numberof0 = COUNT((planes(:,:)==0))
ENDDO
! ****************************************************************************
  WRITE(*,*)
  WRITE(*,*)'***********************************************'
  WRITE(*,*)
  delta = 1
  iter1 = 0
DO WHILE (delta .NE. 0)
  iter1 = iter1 + 1
  WRITE(*,*)
  DO nk = 0,nks-1
    IF (COUNT((planes(nk,:)==0))==0) CYCLE
    DO banda = 1,nbands
      IF (planes(nk,banda) == 0) THEN
!        WRITE(*,*) ' Free place at plane ',banda,n0(nk),n1(nk),n2(nk),n3(nk)
 
        IF (n0(nk) > 0) THEN
          IF (planes(n0(nk),banda) > 0) THEN
          IF (connections2(n0(nk),planes(n0(nk),banda),2) > 0) THEN
            IF (ANY(planes(nk,:) .NE. connections2(n0(nk),planes(n0(nk),banda),2))) THEN  ! Avoid repetitions
              e0 = eigenvalues(n0(nk),planes(n0(nk),banda)) - tol
              e1 = eigenvalues(n0(nk),planes(n0(nk),banda)) + tol
              IF (eigenvalues(nk,connections2(n0(nk),planes(n0(nk),banda),2)) > e0 .AND. &
                  eigenvalues(nk,connections2(n0(nk),planes(n0(nk),banda),2)) < e1) THEN
      
                planes(nk,banda) = connections2(n0(nk),planes(n0(nk),banda),2)
                quality(nk,banda) = 3
                WRITE(*,*) ' Value in 0- ',n0(nk),banda,' Choose: ', planes(nk,banda)
              ENDIF
            ENDIF
          ENDIF
          ENDIF
        ENDIF
 
        IF (n1(nk) > 0) THEN
          IF (planes(n1(nk),banda) > 0) THEN
          IF (connections2(n1(nk),planes(n1(nk),banda),3) > 0) THEN
            IF (ANY(planes(nk,:) .NE. connections2(n1(nk),planes(n1(nk),banda),3))) THEN  ! Avoid repetitions
              e0 = eigenvalues(n1(nk),planes(n1(nk),banda)) - tol
              e1 = eigenvalues(n1(nk),planes(n1(nk),banda)) + tol
              IF (eigenvalues(nk,connections2(n1(nk),planes(n1(nk),banda),3)) > e0 .AND. &
                  eigenvalues(nk,connections2(n1(nk),planes(n1(nk),banda),3)) < e1) THEN
      
                planes(nk,banda) = connections2(n1(nk),planes(n1(nk),banda),3)
                quality(nk,banda) = 3
                WRITE(*,*) ' Value in 1- ',n1(nk),banda,' choose: ', planes(nk,banda)
              ENDIF
            ENDIF
          ENDIF
          ENDIF
        ENDIF
 
        IF (n2(nk) > 0) THEN
          IF (planes(n2(nk),banda) > 0) THEN
          IF (connections2(n2(nk),planes(n2(nk),banda),0) > 0) THEN
            IF (ANY(planes(nk,:) .NE. connections2(n2(nk),planes(n2(nk),banda),0))) THEN  ! Avoid repetitions
              e0 = eigenvalues(n2(nk),planes(n2(nk),banda)) - tol
              e1 = eigenvalues(n2(nk),planes(n2(nk),banda)) + tol
              IF (eigenvalues(nk,connections2(n2(nk),planes(n2(nk),banda),0)) > e0 .AND. &
                  eigenvalues(nk,connections2(n2(nk),planes(n2(nk),banda),0)) < e1) THEN
      
                planes(nk,banda) = connections2(n2(nk),planes(n2(nk),banda),0)
                quality(nk,banda) = 3
                WRITE(*,*) ' Value in 2- ',n2(nk),banda,' choose: ', planes(nk,banda)
              ENDIF
            ENDIF
          ENDIF
          ENDIF
        ENDIF
 
        IF (n3(nk) > 0) THEN
          IF (planes(n3(nk),banda) > 0) THEN
          IF (connections2(n3(nk),planes(n3(nk),banda),1) > 0) THEN
            IF (ANY(planes(nk,:) .NE. connections2(n3(nk),planes(n3(nk),banda),1))) THEN  ! Avoid repetitions
              e0 = eigenvalues(n3(nk),planes(n3(nk),banda)) - tol
              e1 = eigenvalues(n3(nk),planes(n3(nk),banda)) + tol
              IF (eigenvalues(nk,connections2(n3(nk),planes(n3(nk),banda),1)) > e0 .AND. &
                  eigenvalues(nk,connections2(n3(nk),planes(n3(nk),banda),1)) < e1) THEN
      
                planes(nk,banda) = connections2(n3(nk),planes(n3(nk),banda),1)
                quality(nk,banda) = 3
                WRITE(*,*) ' Value in 3- ',n3(nk),banda,' choose: ', planes(nk,banda)
              ENDIF
            ENDIF
          ENDIF
          ENDIF
        ENDIF
 
      ENDIF
    ENDDO
  ENDDO
  delta = numberof0 - COUNT((planes(:,:)==0))
  numberof0 = COUNT((planes(:,:)==0))
ENDDO

! ****************************************************************************
! solving a few more points:

  CALL eig_compare

  CALL eig_compare

! ****************************************************************************
  WRITE(*,*)
  WRITE(*,*) ' Final number of zeros:',numberof0
  WRITE(*,*) ' Number of iterations to achieve convergence: ',iter,iter1
  WRITE(*,*)
! ****************************************************************************

  DO i = 1,nbands
    WRITE(*,*)' Band ',i,' has ',COUNT((planes(:,i)==0)),' zeros.'
  ENDDO
  WRITE(*,*)




! ****************************************************************************
! Write final results
  OPEN(UNIT=3,FILE='apontador',STATUS='UNKNOWN')

  WRITE(*,*)
  nk = -1
  DO j = 0,numero_ky-1
    WRITE(*,*)
    DO i = 0,numero_kx-1
      nk = nk + 1
      WRITE(*,fmt2,advance="no") nk
    ENDDO
    WRITE(*,*)
  ENDDO
  WRITE(*,*)

  DO banda = 1,nbands
    nk = -1
    WRITE(*,*)
    WRITE(*,*)banda, '         | y  x ->'
    DO j = 0,numero_ky-1
      WRITE(*,*) 
      DO i = 0,numero_kx-1
        nk = nk + 1
        WRITE(*,fmt2,advance="no") planes(nk,banda)                       ! prints attributed bands to output
        WRITE(3,fmt3) nk, planes(nk,banda), banda, quality(nk,banda)      ! Writes apontador to file
      ENDDO
      WRITE(*,*)
    ENDDO
  ENDDO

  CLOSE(UNIT=3)

  DO banda = 1,nbands
    nk = -1
    WRITE(*,*)
    WRITE(*,*)banda, ' Quality  | y  x ->',COUNT((quality(:,banda)==0)),&
           COUNT((quality(:,banda)==1)),COUNT((quality(:,banda)==2)),COUNT((quality(:,banda)==3)), &
           COUNT((quality(:,banda)==11)),COUNT((quality(:,banda)==12))
    DO j = 0,numero_ky-1
      WRITE(*,*)
      DO i = 0,numero_kx-1
        nk = nk + 1
        WRITE(*,fmt2,advance="no") quality(nk,banda)    ! prints attributed bands to output
      ENDDO
      WRITE(*,*)
    ENDDO
  ENDDO
  WRITE(*,*)


END PROGRAM compara





