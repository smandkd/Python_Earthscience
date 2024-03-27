program main
        implicit none
        double precision :: x1, x2, x, e

        interface
                double precision function f(x)
                        double precision, intent(in) :: x
                end function f
        end interface

        write(*, *) '초기값 x1, x2=' ! 6: console, monitor
        read(*, *) x1, x2            ! 5: console, monitor
        write(*, *) '수렴 판정 조건='
        read(*, *) e

        do while ( abs(x1-x2) >= e)
                x=(x1+x2)/2.0

                if(f(x1)*f(x) < 0) then
                        x2=x
                else
                        x1=x
                endif
        end do

        write(*, '(A, F8.5)') '답 x=', x
        stop
end program main

double precision function f(x)
        implicit none
        double precision, intent(in) :: x

        f=x-exp(-x)
end function f