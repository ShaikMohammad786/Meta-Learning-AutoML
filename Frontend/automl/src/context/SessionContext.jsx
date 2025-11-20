import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import {
  fetchProfile as fetchProfileApi,
  clearTokenStorage,
  persistToken,
} from '../lib/api'

const SessionContext = createContext({
  token: null,
  setToken: () => {},
  profile: null,
  setProfile: () => {},
  loadingProfile: false,
  logout: () => {},
})

export const SessionProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem('metaml_token'))
  const [profile, setProfile] = useState(null)
  const [loadingProfile, setLoadingProfile] = useState(false)

  useEffect(() => {
    if (token) {
      persistToken(token)
    } else {
      clearTokenStorage()
    }
  }, [token])

  useEffect(() => {
    if (!token) {
      setProfile(null)
      return
    }

    let ignore = false
    setLoadingProfile(true)

    fetchProfileApi(token)
      .then((data) => {
        if (!ignore) setProfile(data.user)
      })
      .catch(() => {
        if (!ignore) {
          setProfile(null)
          setToken(null)
        }
      })
      .finally(() => {
        if (!ignore) setLoadingProfile(false)
      })

    return () => {
      ignore = true
    }
  }, [token])

  const logout = () => {
    setToken(null)
    setProfile(null)
  }

  const value = useMemo(
    () => ({
      token,
      setToken,
      profile,
      setProfile,
      loadingProfile,
      logout,
    }),
    [token, profile, loadingProfile],
  )

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export const useSession = () => useContext(SessionContext)

