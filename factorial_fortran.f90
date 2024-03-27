program compute_factorial

        implicit none
        integer :: n

        interface
                function recur_factorial(n) result(facto)
                        integer :: facto
                        integer, intent(in) :: n
                end function recur_factorial
        end interface

        write(*, '(A)', advance = "no") "Enter n for computing n!: "
        read(*, *) n

        write(*, 100) n, "factorial is ", recur_factorial(n)
        100 format (I3, 2x, A, 2x, I12)

end program compute_factorial

recursive function recur_factorial(n) result(facto)
        implicit none
        integer :: facto
        integer, intent(in) :: n

        if(n == 0) then
                facto = 1
        else
                facto = n * recur_factorial(n-1)
        end if

end function recur_factorial