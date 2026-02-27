import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

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

    const Profile: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
        return (
            <div class={classNames(displayClass, "profile-card")}>
                <div class="profile-avatar-container">
                    <img src={opts.avatar} alt="Avatar" class="profile-avatar" />
                </div>
                <h2 class="profile-name">{opts.name}</h2>
                <p class="profile-title">{opts.title}</p>
                <p class="profile-description">{opts.description}</p>

                <div class="profile-social">
                    {opts.github && (
                        <a href={opts.github} target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.02c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                            </svg>
                        </a>
                    )}
                    {opts.bilibili && (
                        <a href={opts.bilibili} target="_blank" rel="noopener noreferrer" aria-label="Bilibili">
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
                        <a href={opts.mail} target="_blank" rel="noopener noreferrer" aria-label="Mail">
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
  }

  .profile-social a:hover {
    background-color: var(--secondary);
    color: var(--light);
    transform: translateY(-2px);
  }
  `

    return Profile
}) satisfies QuartzComponentConstructor
