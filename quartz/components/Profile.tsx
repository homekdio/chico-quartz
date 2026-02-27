import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"

interface ProfileOptions {
  name: string
  avatar: string
  title: string
  description: string
  github?: string
  bilibili?: string
  mail?: string
}

const defaultOptions: ProfileOptions = {
  name: "Chico",
  avatar: "/static/InfoPic.jpg", // 头像地址
  title: "Coder", // 职位
  description: "一个喜欢分享知识的人", // 介绍
  github: "https://github.com//homekdio", // GitHub地址
  bilibili: "https://www.bilibili.com/", // Bilibili地址
  mail: "mailto:csqichao@qq.com", // 邮箱地址
}

export default ((userOpts?: Partial<ProfileOptions>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const Profile: QuartzComponent = ({ displayClass, fileData }: QuartzComponentProps) => {
    const avatarPath = opts.avatar.startsWith("/")
      ? resolveRelative(fileData.slug!, opts.avatar.slice(1) as FullSlug)
      : opts.avatar

    return (
      <div class={classNames(displayClass, "profile-card")}>
        <div class="profile-avatar-container">
          <img src={avatarPath} alt="Avatar" class="profile-avatar" />
        </div>
        <div class="profile-text">
          <h2 class="profile-name">{opts.name}</h2>
          <p class="profile-title">{opts.title}</p>
        </div>
        <p class={classNames("profile-description", "desktop-only")}>{opts.description}</p>

        <div class={classNames("profile-social", "desktop-only")}>
          {opts.github && (
            <a href={opts.github} target="_blank" rel="noopener noreferrer" aria-label="GitHub" data-tooltip="GitHub">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.02c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
              </svg>
            </a>
          )}
          {opts.bilibili && (
            <a href={opts.bilibili} target="_blank" rel="noopener noreferrer" aria-label="Bilibili" data-tooltip="Bilibili">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="6" width="18" height="13" rx="2" ry="2"></rect>
                <path d="M7 6l2-3"></path>
                <path d="M17 6l-2-3"></path>
                <path d="M9 11v3"></path>
                <path d="M15 11v3"></path>
              </svg>
            </a>
          )}
          {opts.mail && (
            <a href={opts.mail} target="_blank" rel="noopener noreferrer" aria-label="Mail" data-tooltip="邮箱">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="4" width="20" height="16" rx="2" ry="2"></rect>
                <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"></path>
              </svg>
            </a>
          )}
        </div>
      </div>
    )
  }

  Profile.css = `
  .profile-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--light);
    border: 1px solid var(--lightgray);
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  }

  .profile-avatar-container {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    overflow: hidden;
    margin-bottom: 0.3rem;
    border: 3px solid var(--secondary);
  }

  .profile-avatar {
    width: 100%;
    height: 100%;
    object-fit: cover;
    margin: 0;
  }

  .profile-card h2.profile-name {
    font-size: 1.1rem;
    margin: 0 0 0.1rem 0;
    color: var(--dark);
    font-weight: 700;
  }

  .profile-title {
    font-size: 0.8rem;
    color: var(--tertiary);
    margin: 0 0 0.5rem 0;
    font-family: var(--codeFont);
  }

  .profile-description {
    font-size: 0.75rem;
    color: var(--gray);
    text-align: center;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
  }

  .profile-social {
    display: flex;
    gap: 0.8rem;
    justify-content: center;
  }

  .profile-social a {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background-color: var(--lightgray);
    color: var(--darkgray);
    transition: all 0.2s ease;
    position: relative;
  }

  .profile-social a:hover {
    background-color: var(--secondary);
    color: var(--light);
    transform: translateY(-2px);
  }

  /* Tooltip 样式 */
  .profile-social a::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(0);
    padding: 4px 8px;
    background-color: var(--dark);
    color: var(--light);
    font-size: 10px;
    border-radius: 4px;
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: all 0.2s ease;
    pointer-events: none;
    margin-bottom: 8px;
    z-index: 100;
  }

  .profile-social a::before {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 4px solid transparent;
    border-top-color: var(--dark);
    opacity: 0;
    visibility: hidden;
    transition: all 0.2s ease;
    pointer-events: none;
    margin-bottom: 0px;
    z-index: 100;
  }

  .profile-social a:hover::after {
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(-2px);
  }

  .profile-social a:hover::before {
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(-2px);
  }

  @media all and (max-width: 800px) {
    .profile-card {
      flex-direction: row;
      align-items: center;
      padding: 0;
      margin-bottom: 0;
      border: none;
      box-shadow: none;
      background-color: transparent;
      gap: 0.8rem;
    }

    .profile-avatar-container {
      width: 40px;
      height: 40px;
      margin-bottom: 0;
      border-width: 2px;
    }

    .profile-text {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      justify-content: center;
    }
    
    .profile-card h2.profile-name {
      font-size: 1.1rem;
      margin: 0;
      line-height: 1.2;
    }
    
    .profile-title {
      font-size: 0.8rem;
      margin: 0;
      line-height: 1.2;
    }
  }
  `

  return Profile
}) satisfies QuartzComponentConstructor
